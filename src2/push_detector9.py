import cv2
import numpy as np
import time
import json
import sqlite3
import csv
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import math

from video_handler import VideoHandler
from terminal_boundary_drawer import TerminalBoundaryDrawer
from connector_detector import ConnectorDetector
from test_1_arduino import ArduinoController
from postgres_database_manager import PostgresDatabaseManager

@dataclass
class TrackedConnector:
    """Simplified tracked connector for robust push detection"""
    id: int
    current_centroid: Tuple[int, int]
    previous_centroids: List[Tuple[int, int]]
    kalman_filter: cv2.KalmanFilter
    last_seen_frame: int
    age: int
    confidence_history: List[float]
    area_history: List[float]
    
    # Robust push detection
    assigned_terminal: Optional[int]  # Only interact with closest terminal
    push_sessions: Dict[int, List[Dict[str, Any]]]  # terminal_id -> session data
    current_session: Optional[Dict[str, Any]]

@dataclass
class PushSession:
    """A single push session for a connector-terminal pair"""
    connector_id: int
    terminal_id: int
    start_frame: int
    push_events: List[Dict[str, Any]]  # List of individual pushes
    last_push_frame: Optional[int]
    status: str  # 'active', 'waiting_second', 'completed', 'failed'

@dataclass
class PushEvent:
    """Individual push event within a session"""
    connector_id: int
    terminal_id: int
    push_number: int  # 1 or 2
    start_frame: int
    end_frame: Optional[int]
    max_depth: float
    duration_frames: int
    timestamp: float

class SimpleDatabaseManager:
    """
    Simplified database manager for storing essential terminal status data with CSV export
    """
    
    def __init__(self, db_path: str = "terminal_status.db"):
        """
        Initialize simplified database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database and create simplified table"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            
            # Create simplified evaluations table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    terminal_statuses TEXT NOT NULL
                )
            """)
            
            # Create index for faster queries
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp 
                ON evaluations (timestamp)
            """)
            
            self.connection.commit()
            print(f"✅ Database initialized: {self.db_path}")
            
        except sqlite3.Error as e:
            print(f"❌ Database initialization error: {e}")
            self.connection = None
    
    def save_evaluation(self, terminals: List[Dict[str, Any]]) -> Optional[int]:
        """
        Save simplified evaluation data to database (APPEND mode)
        
        Args:
            terminals: List of terminal configurations with current status
            
        Returns:
            Evaluation ID if successful, None if failed
        """
        if not self.connection:
            print("❌ Database not available")
            return None
        
        try:
            current_time = datetime.now()
            timestamp = current_time.isoformat()
            
            # Determine overall status
            has_wait = False
            has_fail = False
            
            for terminal in terminals:
                color = terminal.get('color', (0, 165, 255))
                
                # Check by color (more reliable)
                if color in [(0, 165, 255), (0, 255, 255)]:  # Orange or Yellow (waiting)
                    has_wait = True
                elif color == (0, 0, 255):  # Red (failed)
                    has_fail = True
            
            # Determine overall status: WAIT > NG > OK
            if has_wait:
                overall_status = "WAIT"
            elif has_fail:
                overall_status = "NG"
            else:
                overall_status = "OK"
            
            # Create simplified terminal status string
            terminal_status_list = []
            for i, terminal in enumerate(terminals):
                color = terminal.get('color', (0, 165, 255))
                
                # Convert to simple status
                if color == (0, 255, 0):  # Green
                    simple_status = "OK"
                elif color == (0, 0, 255):  # Red
                    simple_status = "NG"
                else:  # Orange or Yellow
                    simple_status = "WAIT"
                
                terminal_status_list.append(f"T{i+1}:{simple_status}")
            
            terminal_statuses = ",".join(terminal_status_list)
            
            # Insert evaluation record (APPEND - no deletion of previous data)
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO evaluations (timestamp, overall_status, terminal_statuses)
                VALUES (?, ?, ?)
            """, (timestamp, overall_status, terminal_statuses))
            
            evaluation_id = cursor.lastrowid
            self.connection.commit()
            
            print(f"✅ Evaluation saved to database (ID: {evaluation_id}):")
            print(f"   📅 Timestamp: {timestamp}")
            print(f"   🎯 Overall Status: {overall_status}")
            print(f"   📋 Terminal Statuses: {terminal_statuses}")
            
            return evaluation_id
            
        except sqlite3.Error as e:
            print(f"❌ Database save error: {e}")
            self.connection.rollback()
            return None
    
    def export_to_csv(self, csv_path: str, limit: Optional[int] = None) -> bool:
        """
        Export evaluation data to CSV file
        
        Args:
            csv_path: Path where CSV file will be saved
            limit: Optional limit on number of records (None = all records)
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.connection:
            print("❌ Database not available for export")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Query data
            if limit:
                cursor.execute("""
                    SELECT id, timestamp, overall_status, terminal_statuses
                    FROM evaluations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id, timestamp, overall_status, terminal_statuses
                    FROM evaluations 
                    ORDER BY timestamp ASC
                """)
            
            rows = cursor.fetchall()
            
            if not rows:
                print("❌ No data found to export")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
            
            # Write to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['ID', 'Timestamp', 'Overall_Status', 'Terminal_Statuses'])
                
                # Write data rows
                for row in rows:
                    writer.writerow(row)
            
            print(f"✅ Successfully exported {len(rows)} records to: {csv_path}")
            print(f"   📊 File size: {os.path.getsize(csv_path)} bytes")
            
            return True
            
        except Exception as e:
            print(f"❌ CSV export error: {e}")
            return False
    
    def export_to_csv_detailed(self, csv_path: str, limit: Optional[int] = None) -> bool:
        """
        Export evaluation data to CSV with individual terminal columns
        
        Args:
            csv_path: Path where CSV file will be saved
            limit: Optional limit on number of records (None = all records)
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.connection:
            print("❌ Database not available for export")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Query data
            if limit:
                cursor.execute("""
                    SELECT id, timestamp, overall_status, terminal_statuses
                    FROM evaluations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id, timestamp, overall_status, terminal_statuses
                    FROM evaluations 
                    ORDER BY timestamp ASC
                """)
            
            rows = cursor.fetchall()
            
            if not rows:
                print("❌ No data found to export")
                return False
            
            # Determine max number of terminals
            max_terminals = 0
            for row in rows:
                terminal_statuses = row[3]  # terminal_statuses column
                terminal_count = len(terminal_statuses.split(','))
                max_terminals = max(max_terminals, terminal_count)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
            
            # Write to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header with individual terminal columns
                header = ['ID', 'Timestamp', 'Overall_Status']
                for i in range(max_terminals):
                    header.append(f'Terminal_{i+1}')
                writer.writerow(header)
                
                # Write data rows
                for row in rows:
                    csv_row = [row[0], row[1], row[2]]  # ID, Timestamp, Overall_Status
                    
                    # Parse terminal statuses
                    terminal_statuses = row[3]
                    terminal_dict = {}
                    for term_status in terminal_statuses.split(','):
                        if ':' in term_status:
                            term_name, status = term_status.split(':')
                            terminal_dict[term_name] = status
                    
                    # Add terminal statuses to row
                    for i in range(max_terminals):
                        term_key = f'T{i+1}'
                        csv_row.append(terminal_dict.get(term_key, ''))
                    
                    writer.writerow(csv_row)
            
            print(f"✅ Successfully exported {len(rows)} records to detailed CSV: {csv_path}")
            print(f"   📊 File size: {os.path.getsize(csv_path)} bytes")
            print(f"   🏗️  Format: Individual columns for {max_terminals} terminals")
            
            return True
            
        except Exception as e:
            print(f"❌ Detailed CSV export error: {e}")
            return False
    
    def get_record_count(self) -> int:
        """Get total number of records in database"""
        if not self.connection:
            return 0
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            count = cursor.fetchone()[0]
            return count
        except sqlite3.Error:
            return 0
    
    def get_recent_evaluations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent evaluation records
        
        Args:
            limit: Maximum number of evaluations to retrieve
            
        Returns:
            List of evaluation dictionaries
        """
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, timestamp, overall_status, terminal_statuses
                FROM evaluations 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            evaluations = []
            for row in cursor.fetchall():
                evaluations.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'overall_status': row[2],
                    'terminal_statuses': row[3]
                })
            
            return evaluations
            
        except sqlite3.Error as e:
            print(f"❌ Database query error: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get simple statistics from all evaluations
        
        Returns:
            Dictionary with basic statistical data
        """
        if not self.connection:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Overall evaluation stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    SUM(CASE WHEN overall_status = 'OK' THEN 1 ELSE 0 END) as ok_evaluations,
                    SUM(CASE WHEN overall_status = 'NG' THEN 1 ELSE 0 END) as ng_evaluations,
                    SUM(CASE WHEN overall_status = 'WAIT' THEN 1 ELSE 0 END) as wait_evaluations
                FROM evaluations
            """)
            
            stats_row = cursor.fetchone()
            
            # Latest evaluation
            cursor.execute("""
                SELECT timestamp, overall_status, terminal_statuses
                FROM evaluations 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            latest_evaluation = cursor.fetchone()
            
            return {
                'total_evaluations': stats_row[0] or 0,
                'ok_evaluations': stats_row[1] or 0,
                'ng_evaluations': stats_row[2] or 0,
                'wait_evaluations': stats_row[3] or 0,
                'latest_evaluation': latest_evaluation
            }
            
        except sqlite3.Error as e:
            print(f"❌ Database statistics error: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("🔒 Database connection closed")

class RobustConnectorTracker:
    """
    Robust connector tracking with proximity-based push detection
    Eliminates false positives from jittery paths and distant terminals
    """
    
    def __init__(self, max_disappeared: int = 30, 
                 proximity_threshold: float = 80.0,    # Must be within 80px of terminal
                 push_depth_threshold: float = 15.0,   # Depth to consider as push
                 min_push_duration: int = 8,           # Minimum frames for valid push
                 second_push_timeout: int = 150,       # 5 seconds at 30fps
                 depth_smoothing_window: int = 5,      # Smooth depth over 5 frames
                 result_rectangle_width: int = 140,    # Width of result rectangle around terminal
                 result_rectangle_height: int = 100,   # Height of result rectangle around terminal
                 arduino_port: str = 'COM3'):          # 🆕 Arduino COM port
        """
        Initialize robust tracking system
        
        Args:
            proximity_threshold: Max distance to assign terminal to connector
            push_depth_threshold: Minimum depth for valid push
            min_push_duration: Minimum frames to sustain push
            second_push_timeout: Max frames between first and second push
            depth_smoothing_window: Frames to smooth depth measurements
            result_rectangle_width: Width of larger rectangle around terminal for OK/NG display
            result_rectangle_height: Height of larger rectangle around terminal for OK/NG display
            arduino_port: COM port for Arduino communication (e.g., 'COM3')
        """
        self.max_disappeared = max_disappeared
        self.proximity_threshold = proximity_threshold
        self.push_depth_threshold = push_depth_threshold
        self.min_push_duration = min_push_duration
        self.second_push_timeout = second_push_timeout
        self.depth_smoothing_window = depth_smoothing_window
        self.result_rectangle_width = result_rectangle_width
        self.result_rectangle_height = result_rectangle_height
        
        # Tracking state
        self.tracked_connectors: Dict[int, TrackedConnector] = {}
        self.next_connector_id = 0
        self.frame_count = 0
        
        # Push detection state
        self.push_sessions: Dict[Tuple[int, int], PushSession] = {}  # (connector_id, terminal_id) -> session
        self.push_events: List[PushEvent] = []
        self.terminal_states: Dict[int, Dict[str, Any]] = {}
        
        # Terminal centers cache
        self.terminal_centers: Dict[int, Tuple[int, int]] = {}
        
        # Performance tracking
        self.tracking_stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'push_events': 0,
            'successful_insertions': 0,
            'proximity_rejections': 0,
            'depth_rejections': 0,
            'duration_rejections': 0
        }
        
        # Terminal status locking system
        self.terminal_status_locked = {}  # terminal_id -> bool (locked status)
        self.evaluation_results = []  # Store evaluation history
        
        # Auto-reset system
        self.auto_reset_timer_start = None  # Time when all terminals completed
        self.auto_reset_delay = 10.0  # 10 seconds delay
        self.auto_reset_triggered = False  # Flag to prevent multiple auto-resets
        
        # 🆕 Database integration with PostgreSQL and CSV export
        self.db_manager = PostgresDatabaseManager()
        
        # 🆕 Arduino controller integration
        self.arduino_controller = ArduinoController(arduino_port)
        self.arduino_connected = False
        self.arduino_port = arduino_port
        self._initialize_arduino()
    
    def _initialize_arduino(self):
        """Initialize Arduino connection for signal transmission"""
        try:
            print(f"🔌 Initializing Arduino connection on {self.arduino_port}...")
            if self.arduino_controller.connect():
                self.arduino_connected = True
                print("✅ Arduino controller integrated successfully")
                # Start response monitoring
                self.arduino_controller.start_response_monitor()
            else:
                self.arduino_connected = False
                print("⚠️  Arduino controller not available - signals will be logged only")
        except Exception as e:
            print(f"⚠️  Arduino initialization failed: {e}")
            self.arduino_connected = False
    
    def close_arduino_connection(self):
        """Close Arduino connection when shutting down"""
        if hasattr(self, 'arduino_controller') and self.arduino_connected:
            try:
                self.arduino_controller.disconnect()
                print("🔌 Arduino connection closed")
            except Exception as e:
                print(f"⚠️  Error closing Arduino connection: {e}")
    
    def _create_kalman_filter(self, initial_pos: Tuple[int, int]) -> cv2.KalmanFilter:
        """Create Kalman filter for position tracking"""
        kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Noise covariance matrices
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initial state
        kf.statePre = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        kf.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        
        return kf
    
    def _draw_result_rectangle(self, frame: np.ndarray, terminal_center: Tuple[int, int], 
                              status: str, color: Tuple[int, int, int]):
        """
        Draw a larger rectangle around the terminal center showing OK/NG result
        
        Args:
            frame: Image frame to draw on
            terminal_center: Center coordinates of the terminal
            status: Terminal status ('success' or 'failed')
            color: BGR color tuple for the terminal
        """
        # Only draw result rectangle for final states (success/failed)
        if status not in ['success', 'failed']:
            return
        
        # Calculate rectangle bounds using separate width and height
        half_width = self.result_rectangle_width // 2
        half_height = self.result_rectangle_height // 2
        x1 = terminal_center[0] - half_width
        y1 = terminal_center[1] - half_height
        x2 = terminal_center[0] + half_width
        y2 = terminal_center[1] + half_height
        
        # Ensure rectangle stays within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width - 1, x2)
        y2 = min(frame_height - 1, y2)
        
        # Determine result text and colors based on status
        if status == 'success' and color == (0, 255, 0):  # Green = SUCCESS
            result_text = "OK"
            rect_color = (0, 255, 0)  # Green
            text_color = (255, 255, 255)  # White text
        elif status == 'failed' and color == (0, 0, 255):  # Red = FAILED
            result_text = "NG"
            rect_color = (0, 0, 255)  # Red
            text_color = (255, 255, 255)  # White text
        else:
            return  # Don't draw if status doesn't match expected final states
        
        # Draw the larger rectangle outline (thick border)
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 6)
        
        # Draw a semi-transparent filled rectangle for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), rect_color, -1)
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        
        # Calculate text size and position for centering
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(result_text, font, font_scale, thickness)
        
        # Center the text in the rectangle
        text_x = terminal_center[0] - text_width // 2
        text_y = terminal_center[1] + text_height // 2
        
        # Draw text with black outline for better visibility
        cv2.putText(frame, result_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, result_text, (text_x, text_y), font, font_scale, text_color, thickness)

    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_terminal_center(self, terminal_points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate center of terminal polygon"""
        if not terminal_points:
            return (0, 0)
        
        center_x = int(sum(p[0] for p in terminal_points) / len(terminal_points))
        center_y = int(sum(p[1] for p in terminal_points) / len(terminal_points))
        return (center_x, center_y)
    
    def _update_terminal_centers(self, terminals: List[Dict[str, Any]]):
        """Update cached terminal centers"""
        for i, terminal in enumerate(terminals):
            self.terminal_centers[i] = self._calculate_terminal_center(terminal['points'])
    
    def _find_closest_terminal(self, centroid: Tuple[int, int]) -> Optional[int]:
        """
        Find closest terminal within proximity threshold
        
        Args:
            centroid: Connector centroid position
            
        Returns:
            Terminal index if within threshold, None otherwise
        """
        closest_terminal = None
        min_distance = float('inf')
        
        for terminal_id, terminal_center in self.terminal_centers.items():
            distance = self._calculate_distance(centroid, terminal_center)
            
            if distance < self.proximity_threshold and distance < min_distance:
                min_distance = distance
                closest_terminal = terminal_id
        
        return closest_terminal
    
    def _point_to_terminal_distance(self, point: Tuple[int, int], terminal_points: List[Tuple[int, int]]) -> float:
        """Calculate distance from point to terminal boundary (positive = inside)"""
        if len(terminal_points) < 3:
            return float('-inf')
        
        terminal_poly = np.array(terminal_points, dtype=np.int32)
        result = cv2.pointPolygonTest(terminal_poly, point, True)
        return -result  # Flip sign so positive = inside
    
    def _smooth_depth_measurement(self, track: TrackedConnector, current_depth: float) -> float:
        """Apply smoothing to depth measurements to reduce jitter"""
        # Store depth in area_history temporarily (reusing existing field)
        if not hasattr(track, 'depth_history'):
            track.depth_history = deque(maxlen=self.depth_smoothing_window)
        
        track.depth_history.append(current_depth)
        
        if len(track.depth_history) >= 3:
            # Use median filtering to reduce noise
            sorted_depths = sorted(list(track.depth_history))
            return sorted_depths[len(sorted_depths) // 2]
        else:
            return current_depth
    
    def _associate_detections_to_tracks(self, detections: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Associate detections to existing tracks"""
        if not self.tracked_connectors or not detections:
            return []
        
        associations = []
        used_detections = set()
        
        for track_id, track in self.tracked_connectors.items():
            if track.current_centroid is None:
                continue
                
            best_detection_idx = None
            best_distance = float('inf')
            
            for det_idx, detection in enumerate(detections):
                if det_idx in used_detections or not detection['centroid']:
                    continue
                
                distance = self._calculate_distance(track.current_centroid, detection['centroid'])
                max_distance = 100  # pixels
                
                if distance < max_distance and distance < best_distance:
                    best_distance = distance
                    best_detection_idx = det_idx
            
            if best_detection_idx is not None:
                associations.append((track_id, best_detection_idx))
                used_detections.add(best_detection_idx)
        
        return associations
    
    def _update_track(self, track: TrackedConnector, detection: Dict[str, Any]):
        """Update existing track with new detection"""
        centroid = detection['centroid']
        
        # Update Kalman filter
        measurement = np.array([centroid[0], centroid[1]], dtype=np.float32)
        track.kalman_filter.correct(measurement)
        track.kalman_filter.predict()
        
        # Update track state
        track.previous_centroids.append(track.current_centroid)
        if len(track.previous_centroids) > 10:
            track.previous_centroids.pop(0)
        
        track.current_centroid = centroid
        track.last_seen_frame = self.frame_count
        track.age += 1
        track.confidence_history.append(detection['confidence'])
        track.area_history.append(detection['area'])
        
        # Keep history manageable
        if len(track.confidence_history) > 20:
            track.confidence_history.pop(0)
        if len(track.area_history) > 20:
            track.area_history.pop(0)
    
    def _create_new_track(self, detection: Dict[str, Any]) -> TrackedConnector:
        """Create new track from detection"""
        centroid = detection['centroid']
        
        track = TrackedConnector(
            id=self.next_connector_id,
            current_centroid=centroid,
            previous_centroids=[],
            kalman_filter=self._create_kalman_filter(centroid),
            last_seen_frame=self.frame_count,
            age=0,
            confidence_history=[detection['confidence']],
            area_history=[detection['area']],
            assigned_terminal=None,
            push_sessions={},
            current_session=None
        )
        
        self.next_connector_id += 1
        self.tracking_stats['total_tracks'] += 1
        
        return track
    
    def _analyze_push_interaction(self, track: TrackedConnector, terminals: List[Dict[str, Any]]):
        """
        Robust push analysis with proximity-based terminal assignment
        """
        if not track.current_centroid:
            return
        
        # Step 1: Find closest terminal within proximity threshold
        closest_terminal = self._find_closest_terminal(track.current_centroid)
        
        if closest_terminal is None:
            # No terminal within proximity - reset any active session
            if track.current_session:
                track.current_session = None
                track.assigned_terminal = None
                self.tracking_stats['proximity_rejections'] += 1
            return
        
        # Step 2: Assign terminal if not already assigned or if closer terminal found
        if track.assigned_terminal != closest_terminal:
            track.assigned_terminal = closest_terminal
            track.current_session = None  # Reset session when switching terminals
        
        # Step 3: Calculate depth to assigned terminal only
        terminal_points = terminals[track.assigned_terminal]['points']
        raw_depth = self._point_to_terminal_distance(track.current_centroid, terminal_points)
        smoothed_depth = self._smooth_depth_measurement(track, raw_depth)
        
        # Step 4: Analyze push state for assigned terminal
        self._update_push_session(track, track.assigned_terminal, smoothed_depth, terminals)
    
    def _update_push_session(self, track: TrackedConnector, terminal_id: int, depth: float, terminals: List[Dict[str, Any]]):
        """
        Update push session for connector-terminal pair
        """
        session_key = (track.id, terminal_id)
        
        # Initialize terminal state
        if terminal_id not in self.terminal_states:
            self.terminal_states[terminal_id] = {
                'color': (0, 165, 255),  # Orange
                'status': 'waiting',
                'push_count': 0,
                'last_push_frame': None
            }
        
        # Initialize session if needed
        if track.current_session is None:
            track.current_session = {
                'state': 'outside',  # outside, inside, pushing
                'push_start_frame': None,
                'current_push_depth': 0.0,
                'max_depth': 0.0,
                'push_count': 0,
                'last_push_frame': None
            }
        
        session = track.current_session
        terminal_state = self.terminal_states[terminal_id]
        
        # State machine for robust push detection
        if session['state'] == 'outside':
            if depth > 0:  # Entered terminal
                session['state'] = 'inside'
                session['current_push_depth'] = depth
                session['max_depth'] = depth
                print(f"🔄 Connector {track.id} entered terminal {terminal_id} (depth: {depth:.1f})")
        
        elif session['state'] == 'inside':
            if depth > 0:  # Still inside
                session['current_push_depth'] = depth
                session['max_depth'] = max(session['max_depth'], depth)
                
                # Check if deep enough to start push
                if depth >= self.push_depth_threshold and session['state'] != 'pushing':
                    session['state'] = 'pushing'
                    session['push_start_frame'] = self.frame_count
                    print(f"🎯 Connector {track.id} pushing terminal {terminal_id} (depth: {depth:.1f})")
            
            else:  # Exited without reaching push depth
                session['state'] = 'outside'
                session['current_push_depth'] = 0.0
                session['max_depth'] = 0.0
                self.tracking_stats['depth_rejections'] += 1
        
        elif session['state'] == 'pushing':
            if depth > 0:  # Still inside
                session['current_push_depth'] = depth
                session['max_depth'] = max(session['max_depth'], depth)
                
                # Still above threshold - continue pushing
                if depth >= self.push_depth_threshold:
                    push_duration = self.frame_count - session['push_start_frame']
                    # No additional action needed, just maintain state
                else:
                    # Dropped below threshold - end push attempt
                    push_duration = self.frame_count - session['push_start_frame']
                    completed = push_duration >= self.min_push_duration
                    self._end_push_attempt(track, terminal_id, session, terminals, completed)
            
            else:  # Exited while pushing
                push_duration = self.frame_count - session['push_start_frame']
                completed = push_duration >= self.min_push_duration
                self._end_push_attempt(track, terminal_id, session, terminals, completed)
    
    def _end_push_attempt(self, track: TrackedConnector, terminal_id: int, session: Dict[str, Any], 
                         terminals: List[Dict[str, Any]], completed: bool):
        """
        End a push attempt and determine if it was valid
        """
        # Store push_start_frame before clearing it
        push_start_frame = session['push_start_frame']
        
        if not completed:
            # Push too short - reject
            session['state'] = 'outside'
            session['push_start_frame'] = None
            session['current_push_depth'] = 0.0
            session['max_depth'] = 0.0
            self.tracking_stats['duration_rejections'] += 1
            print(f"❌ Connector {track.id} push too short - rejected")
            return
        
        # Valid push completed
        session['push_count'] += 1
        session['last_push_frame'] = self.frame_count
        session['state'] = 'outside'
        # Clear push_start_frame AFTER using it
        session['current_push_depth'] = 0.0
        
        terminal_state = self.terminal_states[terminal_id]
        
        # Create push event using stored value
        push_event = PushEvent(
            connector_id=track.id,
            terminal_id=terminal_id,
            push_number=session['push_count'],
            start_frame=push_start_frame,  # Use stored value
            end_frame=self.frame_count,
            max_depth=session['max_depth'],
            duration_frames=self.frame_count - push_start_frame if push_start_frame is not None else 0,
            timestamp=time.time()
        )
        
        self.push_events.append(push_event)
        self.tracking_stats['push_events'] += 1
        
        # Update terminal state based on push count
        if session['push_count'] == 1:
            # First push - only update if terminal is not locked
            if not self._is_terminal_locked(terminal_id):
                terminals[terminal_id]['color'] = (0, 255, 255)  # Yellow
                terminals[terminal_id]['status'] = 'first_push'
                terminal_state['push_count'] = 1
                terminal_state['last_push_frame'] = self.frame_count
                
                print(f"🎯 FIRST PUSH: Connector {track.id} → Terminal {terminal_id} (Max depth: {session['max_depth']:.1f}px)")
            else:
                print(f"🔒 Terminal {terminal_id} status already locked - ignoring first push")
        
        elif session['push_count'] == 2:
            # Check timing for second push
            if terminal_state['last_push_frame']:
                frames_since_first = self.frame_count - terminal_state['last_push_frame']
                
                if frames_since_first <= self.second_push_timeout:
                    # Valid second push
                    if not self._is_terminal_locked(terminal_id):
                        terminals[terminal_id]['color'] = (0, 255, 0)  # Green
                        terminals[terminal_id]['status'] = 'success'
                        terminal_state['push_count'] = 2
                        self._lock_terminal_status(terminal_id)
                        
                        self.tracking_stats['successful_insertions'] += 1
                        print(f"✅ SUCCESS: Connector {track.id} → Terminal {terminal_id} (Second push) - STATUS LOCKED")
                    else:
                        print(f"🔒 Terminal {terminal_id} status already locked - ignoring push")
                else:
                    # Timeout - treat as first push of new sequence - only if not locked
                    if not self._is_terminal_locked(terminal_id):
                        terminals[terminal_id]['color'] = (0, 255, 255)  # Yellow
                        terminals[terminal_id]['status'] = 'first_push'
                        terminal_state['push_count'] = 1
                        terminal_state['last_push_frame'] = self.frame_count
                        session['push_count'] = 1  # Reset session count
                        
                        print(f"⏰ TIMEOUT: Terminal {terminal_id} reset, treating as first push")
                    else:
                        print(f"🔒 Terminal {terminal_id} status already locked - ignoring timeout reset")
            else:
                # No previous push recorded - treat as first - only if not locked
                if not self._is_terminal_locked(terminal_id):
                    terminals[terminal_id]['color'] = (0, 255, 255)  # Yellow
                    terminals[terminal_id]['status'] = 'first_push'
                    terminal_state['push_count'] = 1
                    terminal_state['last_push_frame'] = self.frame_count
                    session['push_count'] = 1
                else:
                    print(f"🔒 Terminal {terminal_id} status already locked - ignoring push (no previous record)")
        
        # Reset depth tracking and clear push_start_frame AFTER using it
        session['max_depth'] = 0.0
        session['push_start_frame'] = None
    
    def _check_for_timeouts(self, terminals: List[Dict[str, Any]]):
        """Check for terminals waiting too long for second push"""
        for terminal_id, terminal_state in self.terminal_states.items():
            if (terminal_state['push_count'] == 1 and 
                terminal_state['last_push_frame'] is not None):
                
                frames_since_push = self.frame_count - terminal_state['last_push_frame']
                
                if frames_since_push > self.second_push_timeout:
                    # Timeout - only change status if not locked
                    if not self._is_terminal_locked(terminal_id):
                        terminals[terminal_id]['color'] = (0, 0, 255)  # Red
                        terminals[terminal_id]['status'] = 'failed'
                        terminal_state['push_count'] = 0
                        terminal_state['last_push_frame'] = None
                        self._lock_terminal_status(terminal_id)
                        
                        print(f"❌ TIMEOUT: Terminal {terminal_id} failed (no second push) - STATUS LOCKED")
                    else:
                        print(f"🔒 Terminal {terminal_id} status already locked - ignoring timeout")
    
    def _is_terminal_locked(self, terminal_id: int) -> bool:
        """Check if terminal status is locked"""
        return self.terminal_status_locked.get(terminal_id, False)
    
    def _lock_terminal_status(self, terminal_id: int):
        """Lock terminal status to prevent further changes"""
        self.terminal_status_locked[terminal_id] = True
    
    def _evaluate_all_terminals_and_reset(self, terminals: List[Dict[str, Any]]) -> str:
        """
        Evaluate all terminal statuses and reset system
        
        Returns:
            'OK' if all terminals are green (success)
            'NG' if any terminal is not green
        """
        print("\n" + "="*60)
        print("🔍 TERMINAL STATUS EVALUATION")
        print("="*60)
        
        all_success = True
        evaluation_summary = []
        
        # Check each terminal status
        for i, terminal in enumerate(terminals):
            status = terminal.get('status', 'waiting')
            color_name = self._get_color_name(terminal['color'])
            
            terminal_result = {
                'terminal_id': i,
                'status': status,
                'color': color_name,
                'locked': self._is_terminal_locked(i)
            }
            evaluation_summary.append(terminal_result)
            
            print(f"Terminal {i+1}: {status.upper()} ({color_name}) {'🔒' if self._is_terminal_locked(i) else '🔓'}")
            
            if status != 'success':
                all_success = False
        
        # Determine final result
        final_result = "OK" if all_success else "NG"
        
        print("-" * 60)
        print(f"📊 EVALUATION RESULT: {final_result}")
        print("-" * 60)
        
        # Store evaluation in history
        evaluation_record = {
            'timestamp': time.time(),
            'result': final_result,
            'terminals': evaluation_summary.copy(),
            'total_terminals': len(terminals),
            'successful_terminals': sum(1 for t in evaluation_summary if t['status'] == 'success')
        }
        self.evaluation_results.append(evaluation_record)
        
        # 🆕 SAVE TO DATABASE BEFORE RESET (APPEND MODE)
        print("\n💾 SAVING EVALUATION DATA TO DATABASE...")
        evaluation_id = self.db_manager.save_evaluation(terminals)
        
        if evaluation_id:
            total_records = self.db_manager.get_record_count()
            print(f"✅ Data successfully saved (Total records in DB: {total_records})")
        else:
            print("❌ Failed to save data to database")
        
        # Send signal to terminal (placeholder - replace with actual implementation)
        self._send_signal_to_terminal(final_result)
        
        # Reset all terminals
        self._reset_all_terminals(terminals)
        
        return final_result
    
    def _get_color_name(self, color_bgr: Tuple[int, int, int]) -> str:
        """Convert BGR color tuple to readable name"""
        color_map = {
            (0, 165, 255): "ORANGE",    # Waiting
            (0, 255, 255): "YELLOW",    # First push
            (0, 255, 0): "GREEN",       # Success
            (0, 0, 255): "RED"          # Failed
        }
        return color_map.get(color_bgr, "UNKNOWN")
    
    def _send_signal_to_terminal(self, signal: str):
        """
        Send OK/NG signal to Arduino terminal
        
        Args:
            signal: 'OK' or 'NG' signal to send
        """
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"📡 [{timestamp}] SENDING SIGNAL TO TERMINAL: {signal}")
        
        if self.arduino_connected and self.arduino_controller.connected:
            try:
                # Send signal to Arduino
                success = self.arduino_controller.send_signal(signal)
                if success:
                    print(f"✅ Signal '{signal}' sent to Arduino successfully")
                else:
                    print(f"❌ Failed to send signal '{signal}' to Arduino")
            except Exception as e:
                print(f"❌ Arduino communication error: {e}")
                self.arduino_connected = False
        else:
            print(f"⚠️  Arduino not connected - Signal '{signal}' logged only")
            # Fallback: just log the signal
            print(f"🔗 Signal '{signal}' would be sent to terminal")
    
    def _reset_all_terminals(self, terminals: List[Dict[str, Any]]):
        """Reset all terminals to orange (waiting) status"""
        print("\n🔄 RESETTING ALL TERMINALS...")
        
        for i, terminal in enumerate(terminals):
            terminal['color'] = (0, 165, 255)  # Orange
            terminal['status'] = 'waiting'
        
        # Clear all terminal states and locks
        self.terminal_states.clear()
        self.terminal_status_locked.clear()
        
        # Reset auto-reset system
        self.auto_reset_timer_start = None
        self.auto_reset_triggered = False
        
        # Reset tracking statistics for new evaluation cycle
        self.tracking_stats.update({
            'push_events': 0,
            'successful_insertions': 0,
            'proximity_rejections': 0,
            'depth_rejections': 0,
            'duration_rejections': 0
        })
        
    def _all_terminals_have_final_status(self, terminals: List[Dict[str, Any]]) -> bool:
        """
        Check if all terminals have a final status (success or failed)
        
        Returns:
            True if all terminals have either 'success' or 'failed' status
        """
        for terminal in terminals:
            status = terminal.get('status', 'waiting')
            if status not in ['success', 'failed']:
                return False
        return True
    
    def _check_auto_reset(self, terminals: List[Dict[str, Any]]) -> bool:
        """
        Check if auto-reset should be triggered and handle the timing
        
        Returns:
            True if auto-reset was triggered, False otherwise
        """
        current_time = time.time()
        
        # Check if all terminals have final status
        all_complete = self._all_terminals_have_final_status(terminals)
        
        if all_complete:
            # Start timer if not already started
            if self.auto_reset_timer_start is None:
                self.auto_reset_timer_start = current_time
                print(f"\n⏰ All terminals complete! Auto-reset in {self.auto_reset_delay} seconds...")
                return False
            
            # Check if enough time has passed
            elapsed = current_time - self.auto_reset_timer_start
            if elapsed >= self.auto_reset_delay:
                print(f"\n🔄 AUTO-RESET TRIGGERED after {elapsed:.1f} seconds")
                result = self._evaluate_all_terminals_and_reset(terminals)
                print(f"🎯 AUTO-EVALUATION COMPLETE: {result}")
                # Re-arm timer for next cycle
                self.auto_reset_timer_start = None
                return True
        else:
            # Reset timer if not all terminals are complete anymore
            if self.auto_reset_timer_start is not None:
                self.auto_reset_timer_start = None
        
        return False
    
    def update(self, detections: List[Dict[str, Any]], terminals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main tracking update with robust push detection
        """
        self.frame_count += 1
        
        # Update terminal centers cache
        self._update_terminal_centers(terminals)
        
        # Predict positions
        for track in self.tracked_connectors.values():
            track.kalman_filter.predict()
        
        # Associate detections to tracks
        associations = self._associate_detections_to_tracks(detections)
        
        # Update existing tracks
        updated_track_ids = set()
        for track_id, detection_idx in associations:
            if track_id in self.tracked_connectors:
                self._update_track(self.tracked_connectors[track_id], detections[detection_idx])
                updated_track_ids.add(track_id)
        
        # Create new tracks
        used_detection_indices = {det_idx for _, det_idx in associations}
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detection_indices and detection['centroid']:
                new_track = self._create_new_track(detection)
                self.tracked_connectors[new_track.id] = new_track
                updated_track_ids.add(new_track.id)
        
        # Handle disappeared tracks
        disappeared_tracks = []
        for track_id, track in list(self.tracked_connectors.items()):
            if track_id not in updated_track_ids:
                frames_disappeared = self.frame_count - track.last_seen_frame
                if frames_disappeared > self.max_disappeared:
                    disappeared_tracks.append(track_id)
                    del self.tracked_connectors[track_id]
        
        # Analyze push interactions for all active tracks
        for track in self.tracked_connectors.values():
            self._analyze_push_interaction(track, terminals)
        
        # Check for timeouts
        self._check_for_timeouts(terminals)
        
        # Update statistics
        self.tracking_stats['active_tracks'] = len(self.tracked_connectors)
        
        return {
            'tracked_connectors': list(self.tracked_connectors.values()),
            'new_tracks': [tid for tid in updated_track_ids if tid >= self.next_connector_id - len(detections)],
            'disappeared_tracks': disappeared_tracks,
            'push_events': self.push_events[-5:],
            'terminal_states': self.terminal_states,
            'stats': self.tracking_stats.copy()
        }
    
    def draw_tracking_visualization(self, frame: np.ndarray, tracking_results: Dict[str, Any], 
                                   terminals: List[Dict[str, Any]], debug_mode: bool = True) -> np.ndarray:
        """
        Draw robust tracking visualization
        """
        display_frame = frame.copy()
        
        # Draw terminals
        for i, terminal in enumerate(terminals):
            points = np.array(terminal['points'], np.int32)
            color = terminal['color']
            
            # Draw filled polygon with transparency
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0, display_frame)
            
            # Draw boundary outline
            cv2.polylines(display_frame, [points], True, color, 2)
            
            # Terminal center and proximity circle
            if i in self.terminal_centers:
                center = self.terminal_centers[i]
                
                # Draw center point
                cv2.circle(display_frame, center, 4, (255, 255, 255), -1)
                
                # Draw proximity threshold circle in debug mode
                if debug_mode:
                    cv2.circle(display_frame, center, int(self.proximity_threshold), (128, 128, 128), 1)
                
                # Draw result rectangle for SUCCESS/FAIL terminals
                status = terminal.get('status', 'waiting')
                self._draw_result_rectangle(display_frame, center, status, color)
            
            # Terminal status
            center_x = int(np.mean([p[0] for p in terminal['points']]))
            center_y = int(np.mean([p[1] for p in terminal['points']]))
            
            status = terminal.get('status', 'waiting')
            push_count = self.terminal_states.get(i, {}).get('push_count', 0)
            lock_status = "🔒" if self._is_terminal_locked(i) else "🔓"
            
            label = f"T{i+1}"
            if debug_mode:
                label += f" ({status}:{push_count}){lock_status}"
            
            cv2.putText(display_frame, label, (center_x-20, center_y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw tracked connectors
        for track in tracking_results['tracked_connectors']:
            if not track.current_centroid:
                continue
            
            # Connector centroid
            cv2.circle(display_frame, track.current_centroid, 4, (255, 0, 0), -1)
            cv2.circle(display_frame, track.current_centroid, 8, (255, 0, 0), 2)
            
            # Connection line to assigned terminal
            if debug_mode and track.assigned_terminal is not None and track.assigned_terminal in self.terminal_centers:
                terminal_center = self.terminal_centers[track.assigned_terminal]
                cv2.line(display_frame, track.current_centroid, terminal_center, (0, 255, 255), 1)
            
            # Connector ID and state
            label = f"C{track.id}"
            if debug_mode and track.assigned_terminal is not None:
                session = track.current_session
                state = session['state'] if session else 'none'
                label += f" T{track.assigned_terminal}({state})"
            
            cv2.putText(display_frame, label, 
                       (track.current_centroid[0] - 25, track.current_centroid[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Push session info
            if debug_mode and track.current_session:
                session = track.current_session
                if session['state'] in ['inside', 'pushing']:
                    depth_text = f"D:{session['current_push_depth']:.1f}"
                    cv2.putText(display_frame, depth_text,
                               (track.current_centroid[0] - 20, track.current_centroid[1] + 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                if session['push_count'] > 0:
                    push_text = f"Pushes:{session['push_count']}"
                    cv2.putText(display_frame, push_text,
                               (track.current_centroid[0] - 25, track.current_centroid[1] + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Trail
            if debug_mode and len(track.previous_centroids) > 1:
                for i in range(1, len(track.previous_centroids)):
                    pt1 = track.previous_centroids[i-1]
                    pt2 = track.previous_centroids[i]
                    cv2.line(display_frame, pt1, pt2, (128, 128, 255), 1)
        
        # Terminal Status Table - Always visible (not just in debug mode)
        status_table = self._create_status_table(terminals)
        if status_table is not None:
            # Place status table on TOP RIGHT corner
            frame_width = display_frame.shape[1]
            table_height, table_width = status_table.shape[:2]
            x_offset = frame_width - table_width - 10  # 10px margin from right edge
            y_offset = 10  # 10px margin from top edge
            
            # Ensure the table fits within the frame
            if x_offset >= 0 and y_offset + table_height <= display_frame.shape[0]:
                display_frame[y_offset:y_offset+table_height, x_offset:x_offset+table_width] = status_table

        # Enhanced statistics overlay (moved below status table)
        if debug_mode:
            stats = tracking_results['stats']
            overlay_height = 200  # Increased height for evaluation info
            overlay = np.zeros((overlay_height, 400, 3), dtype=np.uint8)
            
            cv2.putText(overlay, "ROBUST TRACKING STATS", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay, f"Active tracks: {stats['active_tracks']}", (10, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(overlay, f"Push events: {stats['push_events']}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(overlay, f"Successful: {stats['successful_insertions']}", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(overlay, f"Proximity rejections: {stats['proximity_rejections']}", (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(overlay, f"Depth rejections: {stats['depth_rejections']}", (10, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(overlay, f"Duration rejections: {stats['duration_rejections']}", (10, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(overlay, f"Proximity threshold: {self.proximity_threshold}px", (10, 165), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
            
            # Show evaluation history count
            cv2.putText(overlay, f"Evaluations completed: {len(self.evaluation_results)}", (10, 185), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 255, 128), 1)
            
            # Place overlay on frame (TOP RIGHT corner, below status table)
            frame_width = display_frame.shape[1]
            overlay_width = 400
            x_offset = frame_width - overlay_width - 10  # 10px margin from right edge
            
            # Calculate y_offset to place below status table
            status_table_height = 0
            if status_table is not None:
                status_table_height = status_table.shape[0] + 20  # 20px gap
            y_offset_stats = 10 + status_table_height
            
            # Ensure stats overlay fits within frame
            if y_offset_stats + overlay_height <= display_frame.shape[0]:
                display_frame[y_offset_stats:y_offset_stats+overlay_height, x_offset:x_offset+overlay_width] = overlay
        
        return display_frame
    
    def _create_status_table(self, terminals: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Create a status table showing terminal status and overall status
        
        Returns:
            numpy array representing the status table image, or None if no terminals
        """
        if not terminals:
            return None
        
        # Table dimensions (reduced to 2x size)
        num_terminals = len(terminals)
        cell_width = 160  # 80 * 2
        cell_height = 50  # 25 * 2
        header_height = 60  # 30 * 2
        table_width = max(400, cell_width * 2)  # Terminal column + Status column (200 * 2 = 400)
        table_height = header_height + (num_terminals + 2) * cell_height  # +2 for header row and overall status
        
        # Create table background
        table = np.zeros((table_height, table_width, 3), dtype=np.uint8)
        table.fill(40)  # Dark gray background
        
        # Table colors
        header_color = (60, 60, 60)
        border_color = (100, 100, 100)
        text_color = (255, 255, 255)
        
        # Draw header background
        cv2.rectangle(table, (0, 0), (table_width, header_height), header_color, -1)
        
        # Header text (medium font for 2x table)
        cv2.putText(table, "TERMINAL STATUS", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        
        # Draw column headers (medium font and adjusted positions)
        y_pos = header_height + 40
        cv2.putText(table, "Term", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(table, "Status", (cell_width + 20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # Draw horizontal line after column headers (adjusted position)
        y_pos += 10
        cv2.line(table, (10, y_pos), (table_width - 10, y_pos), border_color, 2)
        
        # Terminal status rows
        overall_status = self._calculate_overall_status(terminals)
        
        for i, terminal in enumerate(terminals):
            y_row = header_height + cell_height + (i * cell_height)
            
            # Get terminal status
            terminal_state = self.terminal_states.get(i, {})
            color = terminal.get('color', (0, 165, 255))  # Default orange
            status = terminal.get('status', 'waiting')
            
            # Map status to display text
            status_text = self._get_status_display_text(status, color)
            status_color = self._get_status_text_color(status, color)
            
            # Terminal number (medium font and adjusted position)
            cv2.putText(table, f"T{i+1}", (30, y_row + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            # Status text (medium font and adjusted position)
            cv2.putText(table, status_text, (cell_width + 30, y_row + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Draw row separator (medium thickness line)
            if i < num_terminals - 1:
                cv2.line(table, (10, y_row + cell_height), (table_width - 10, y_row + cell_height), 
                        (60, 60, 60), 1)
        
        # Overall status row (adjusted for 2x table)
        overall_y = header_height + cell_height + (num_terminals * cell_height) + 10
        
        # Draw separator line before overall status (medium thickness line)
        cv2.line(table, (10, overall_y), (table_width - 10, overall_y), border_color, 2)
        
        overall_y += 30
        
        # Overall status (medium font and adjusted positions)
        overall_text = self._get_overall_status_text(overall_status)
        overall_color = self._get_overall_status_color(overall_status)
        
        cv2.putText(table, "Overall:", (30, overall_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(table, overall_text, (cell_width + 30, overall_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, overall_color, 2)
        
        # Draw table border (medium thickness border)
        cv2.rectangle(table, (0, 0), (table_width - 1, table_height - 1), border_color, 2)
        
        # Draw vertical separator between columns (medium thickness line)
        cv2.line(table, (cell_width, header_height), (cell_width, table_height - 1), border_color, 2)
        
        return table
    
    def _get_status_display_text(self, status: str, color: Tuple[int, int, int]) -> str:
        """
        Map terminal status and color to display text
        
        Args:
            status: Terminal status ('waiting', 'success', 'failed', etc.)
            color: BGR color tuple
            
        Returns:
            Display text ('OK', 'NG', 'WAIT')
        """
        # Check color first for more accurate status
        if color == (0, 255, 0):  # Green
            return "OK"
        elif color == (0, 0, 255):  # Red
            return "NG"
        elif color in [(0, 165, 255), (0, 255, 255)]:  # Orange or Yellow
            return "WAIT"
        
        # Fallback to status string
        if status == 'success':
            return "OK"
        elif status == 'failed':
            return "NG"
        else:
            return "WAIT"
    
    def _get_status_text_color(self, status: str, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Get the display color for status text
        
        Args:
            status: Terminal status
            color: BGR color tuple
            
        Returns:
            BGR color for text display
        """
        # Check color first for more accurate status
        if color == (0, 255, 0):  # Green
            return (0, 255, 0)  # Green text
        elif color == (0, 0, 255):  # Red
            return (0, 0, 255)  # Red text
        elif color in [(0, 165, 255), (0, 255, 255)]:  # Orange or Yellow
            return (0, 165, 255)  # Orange text
        
        # Fallback
        return (255, 255, 255)  # White text
    
    def _calculate_overall_status(self, terminals: List[Dict[str, Any]]) -> str:
        """
        Calculate overall status based on all terminal statuses
        
        Args:
            terminals: List of terminal configurations
            
        Returns:
            'WAIT' if any terminal is waiting/orange
            'NG' if any terminal failed/red and none are waiting
            'OK' if all terminals are success/green
        """
        has_wait = False
        has_fail = False
        has_success = False
        
        for i, terminal in enumerate(terminals):
            color = terminal.get('color', (0, 165, 255))
            status = terminal.get('status', 'waiting')
            
            # Check by color (more reliable)
            if color in [(0, 165, 255), (0, 255, 255)]:  # Orange or Yellow (waiting)
                has_wait = True
            elif color == (0, 0, 255):  # Red (failed)
                has_fail = True
            elif color == (0, 255, 0):  # Green (success)
                has_success = True
        
        # Priority: WAIT > NG > OK
        if has_wait:
            return "WAIT"
        elif has_fail:
            return "NG"
        elif has_success:
            return "OK"
        else:
            return "WAIT"  # Default if all terminals are in unknown state
    
    def _get_overall_status_text(self, overall_status: str) -> str:
        """Get display text for overall status"""
        return overall_status
    
    def _get_overall_status_color(self, overall_status: str) -> Tuple[int, int, int]:
        """Get display color for overall status"""
        if overall_status == "OK":
            return (0, 255, 0)  # Green
        elif overall_status == "NG":
            return (0, 0, 255)  # Red
        else:  # WAIT
            return (0, 165, 255)  # Orange

    def export_database_to_csv(self, csv_path: str, detailed: bool = True, limit: Optional[int] = None) -> bool:
        """
        Export database to CSV file
        
        Args:
            csv_path: Path where CSV file will be saved
            detailed: If True, creates individual terminal columns. If False, keeps terminal data as single column
            limit: Optional limit on number of records (None = all records)
            
        Returns:
            True if export successful, False otherwise
        """
        print(f"\n📁 EXPORTING DATABASE TO CSV...")
        print(f"   📝 File path: {csv_path}")
        print(f"   📊 Format: {'Detailed (individual terminal columns)' if detailed else 'Simple (compact format)'}")
        print(f"   📈 Records: {'All records' if limit is None else f'Last {limit} records'}")
        
        if detailed:
            success = self.db_manager.export_to_csv_detailed(csv_path, limit)
        else:
            success = self.db_manager.export_to_csv_simple(csv_path, limit)
        
        if success:
            print(f"✅ CSV export completed successfully!")
            print(f"   📂 File location: {os.path.abspath(csv_path)}")
        else:
            print(f"❌ CSV export failed!")
        
        return success
        """
        Export database to CSV file
        
        Args:
            csv_path: Path where CSV file will be saved
            detailed: If True, creates individual terminal columns. If False, keeps terminal data as single column
            limit: Optional limit on number of records (None = all records)
            
        Returns:
            True if export successful, False otherwise
        """
        print(f"\n📁 EXPORTING DATABASE TO CSV...")
        print(f"   📝 File path: {csv_path}")
        print(f"   📊 Format: {'Detailed (individual terminal columns)' if detailed else 'Simple (compact format)'}")
        print(f"   📈 Records: {'All records' if limit is None else f'Last {limit} records'}")
        
        if detailed:
            success = self.db_manager.export_to_csv_detailed(csv_path, limit)
        else:
            success = self.db_manager.export_to_csv(csv_path, limit)
        
        if success:
            print(f"✅ CSV export completed successfully!")
            print(f"   📂 File location: {os.path.abspath(csv_path)}")
        else:
            print(f"❌ CSV export failed!")
        
        return success

def test_robust_tracking_system(model_path: str, video_source: Optional[str] = None,
                               config_file: str = "terminal_config.json",
                               arduino_port: str = 'COM3'):
    """
    Test the robust proximity-based tracking system with Arduino integration
    """
    print("🎯 ROBUST CONNECTOR TRACKING & PUSH DETECTION TEST")
    print("="*65)
    print("🛡️  Proximity-based push detection enabled")
    print("💾 Database integration with CSV export enabled")
    print("🔌 Arduino integration for OK/NG signal transmission")
    print("   - Proximity threshold: 80.0 pixels (only closest terminal)")
    print("   - Push depth threshold: 15.0 pixels")
    print("   - Minimum push duration: 8 frames (~0.27s)")
    print("   - Second push timeout: 150 frames (5.0s)")
    print("   - Depth smoothing: 5 frame median filter")
    print("   - Database: terminal_status.db (append mode)")
    print("   - CSV Export: On-demand export functionality")
    print(f"   - Arduino port: {arduino_port}")
    
    # Initialize components
    print("\n🚀 Initializing components...")
    
    # Video handler
    video_handler = VideoHandler(video_source, target_fps=30)
    if not video_handler.initialize():
        print("❌ Failed to initialize video")
        return
    
    # Terminal boundary drawer
    boundary_drawer = TerminalBoundaryDrawer(config_file=config_file)
    if not boundary_drawer.load_configuration():
        print("❌ Failed to load terminal configuration")
        return
    
    terminals = boundary_drawer.terminals
    print(f"✅ Loaded {len(terminals)} terminals")
    
    # Connector detector
    detector = ConnectorDetector(model_path, confidence_threshold=0.35, target_classes=['connector'])
    if not detector.initialize_model():
        print("❌ Failed to initialize detector")
        return
    
    # Robust connector tracker with Arduino integration
    tracker = RobustConnectorTracker(
        max_disappeared=30,
        proximity_threshold=40.0,        # Must be within 30px of terminal center
        push_depth_threshold=2.0,       # 2px inside terminal boundary
        min_push_duration=1,             # ~0.27 seconds minimum push
        second_push_timeout=20,         # 5 seconds for second push
        depth_smoothing_window=5,        # Smooth depth over 5 frames
        result_rectangle_width=100,      # Width of OK/NG result rectangle
        result_rectangle_height=175,     # Height of OK/NG result rectangle
        arduino_port=arduino_port        # 🆕 Arduino port configuration
    )
    
    print("✅ Robust tracking system initialized")
    print("💾 Database ready for data storage (append mode)")
    
    # Show initial database status
    total_records = tracker.db_manager.get_record_count()
    print(f"📊 Current database contains {total_records} records")
    
    # Setup display
    cv2.namedWindow('Robust Tracking and Push Detection', cv2.WINDOW_NORMAL)
    
    # Performance tracking
    frame_count = 0
    fps_start_time = time.time()
    debug_mode = True
    paused = False
    
    print("\n🎥 Starting robust tracking test...")
    print("Controls:")
    print("  'd': Toggle debug mode (shows proximity circles & assignments)")
    print("  'p': Pause/resume")
    print("  'r': Reset terminal states (manual reset)")
    print("  't': Print tracking thresholds")
    print("  's': Print current statistics")
    print("  'e': Show detailed push event history")
    print("  'b': Show database statistics")
    print("  'h': Show recent evaluations from database")
    print("  'c': Export database to CSV (detailed format)")
    print("  'v': Export database to CSV (simple format)")
    print("  'n': Export last 10 records to CSV")
    print("  'a': Test Arduino connection (send OK then NG)")
    print("  '+': Increase result rectangle width")
    print("  '-': Decrease result rectangle width")
    print("  'w': Increase result rectangle height")
    print("  'x': Decrease result rectangle height")
    print("  'z': Show current result rectangle dimensions")
    print("\n⏰ AUTO-RESET: System automatically resets 10 seconds after all terminals complete")
    print("💾 AUTO-SAVE: Data automatically saved to database before each reset (append mode)")
    print("📁 CSV EXPORT: Use 'c', 'v', or 'n' keys to export data when needed")
    print("🔌 ARDUINO TEST: Use 'a' key to test Arduino connection manually")
    
    try:
        while True:
            if not paused:
                ret, frame = video_handler.get_frame()
                if not ret:
                    if video_source is not None:
                        print("📹 End of video reached")
                        break
                    else:
                        continue
                
                # Detect connectors
                detections = detector.detect_connectors(frame)
                
                # Update robust tracking
                tracking_results = tracker.update(detections, terminals)
                
                # Check for auto-reset (only when not paused)
                tracker._check_auto_reset(terminals)
                
                frame_count += 1
            
            # Create robust display
            display_frame = tracker.draw_tracking_visualization(
                frame, tracking_results, terminals, debug_mode
            )
            
            # Add frame info
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # FPS display
            if frame_count % 30 == 0 and frame_count > 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed
                fps_start_time = time.time()
            else:
                fps = 0
            
            if fps > 0:
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                           (display_frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mode indicators
            if debug_mode:
                cv2.putText(display_frame, "DEBUG: Proximity analysis ON", 
                           (display_frame.shape[1] - 250, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if paused:
                cv2.putText(display_frame, "PAUSED", 
                           (display_frame.shape[1]//2 - 50, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Auto-reset countdown indicator
            if tracker.auto_reset_timer_start is not None and not tracker.auto_reset_triggered:
                elapsed = time.time() - tracker.auto_reset_timer_start
                remaining = tracker.auto_reset_delay - elapsed
                if remaining > 0:
                    cv2.putText(display_frame, f"AUTO-RESET IN: {remaining:.1f}s", 
                               (display_frame.shape[1]//2 - 120, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Robust Tracking and Push Detection', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
                if debug_mode:
                    print("   - Showing proximity circles around terminals")
                    print("   - Terminal assignments with connection lines")
                    print("   - Push session states and depths")
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
            elif key == ord('r'):
                # Evaluate all terminals and reset system
                result = tracker._evaluate_all_terminals_and_reset(terminals)
                print(f"\n🎯 EVALUATION COMPLETE: {result}")
                print("🔄 System ready for next evaluation cycle")
            elif key == ord('t'):
                print("\n📏 Current Tracking Thresholds:")
                print(f"   - Proximity threshold: {tracker.proximity_threshold} pixels")
                print(f"   - Push depth threshold: {tracker.push_depth_threshold} pixels")
                print(f"   - Minimum push duration: {tracker.min_push_duration} frames")
                print(f"   - Second push timeout: {tracker.second_push_timeout} frames")
                print(f"   - Depth smoothing window: {tracker.depth_smoothing_window} frames")
                print(f"   - Result rectangle dimensions: {tracker.result_rectangle_width}px × {tracker.result_rectangle_height}px")
            elif key == ord('s'):
                stats = tracking_results['stats']
                total_db_records = tracker.db_manager.get_record_count()
                print("\n📊 Current Tracking Statistics:")
                print(f"   - Active tracks: {stats['active_tracks']}")
                print(f"   - Total tracks created: {stats['total_tracks']}")
                print(f"   - Push events: {stats['push_events']}")
                print(f"   - Successful insertions: {stats['successful_insertions']}")
                print(f"   - Proximity rejections: {stats['proximity_rejections']}")
                print(f"   - Depth rejections: {stats['depth_rejections']}")
                print(f"   - Duration rejections: {stats['duration_rejections']}")
                print(f"   - Database records: {total_db_records}")
                
                # Show recent push events with depths
                if tracker.push_events:
                    print(f"\n🎯 Recent Push Events (Last 5):")
                    for i, event in enumerate(tracker.push_events[-5:]):
                        push_type = "1st" if event.push_number == 1 else "2nd"
                        print(f"   {i+1}. Connector {event.connector_id} → Terminal {event.terminal_id}")
                        print(f"      {push_type} push, Max depth: {event.max_depth:.1f}px, Duration: {event.duration_frames} frames")
                else:
                    print(f"\n🎯 No push events recorded yet")
            elif key == ord('e'):
                print("\n📋 Detailed Push Event History:")
                if tracker.push_events:
                    for i, event in enumerate(tracker.push_events):
                        push_type = "1st" if event.push_number == 1 else "2nd"
                        timestamp_str = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
                        print(f"   Event #{i+1}: {timestamp_str}")
                        print(f"      Connector {event.connector_id} → Terminal {event.terminal_id} ({push_type} push)")
                        print(f"      Max depth: {event.max_depth:.1f}px")
                        print(f"      Duration: {event.duration_frames} frames")
                        print(f"      Frames: {event.start_frame} to {event.end_frame}")
                        print()
                else:
                    print("   No push events recorded yet")
            elif key == ord('b'):
                # Show database statistics
                stats = tracker.db_manager.get_statistics()
                total_records = tracker.db_manager.get_record_count()
                print("\n� DATABASE STATISTICS:")
                print(f"   📈 Total Records: {total_records}")
                print(f"   ✅ OK Evaluations: {stats.get('ok_evaluations', 0)}")
                print(f"   ❌ NG Evaluations: {stats.get('ng_evaluations', 0)}")
                print(f"   ⏳ WAIT Evaluations: {stats.get('wait_evaluations', 0)}")
                
                total = stats.get('total_evaluations', 0)
                if total > 0:
                    ok_rate = (stats.get('ok_evaluations', 0) / total) * 100
                    print(f"   🎯 OK Rate: {ok_rate:.1f}%")
                
                if stats.get('latest_evaluation'):
                    latest_timestamp, latest_status, latest_terminals = stats['latest_evaluation']
                    print(f"   🕐 Latest: {latest_timestamp[:19]} ({latest_status}) - {latest_terminals}")
            elif key == ord('h'):
                # Show recent evaluations from database
                recent_evaluations = tracker.db_manager.get_recent_evaluations(limit=5)
                total_records = tracker.db_manager.get_record_count()
                print(f"\n📋 RECENT EVALUATIONS (Showing 5 of {total_records} total records):")
                if recent_evaluations:
                    for i, evaluation in enumerate(recent_evaluations):
                        timestamp = evaluation['timestamp'][:19]  # Remove microseconds
                        status = evaluation['overall_status']
                        terminals = evaluation['terminal_statuses']
                        print(f"   {i+1}. ID:{evaluation['id']} | {timestamp} | {status} | {terminals}")
                else:
                    print("   No evaluations found in database")
            elif key == ord('c'):
                # Export database to CSV (detailed format)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = f"exports/terminal_evaluations_detailed_{timestamp}.csv"
                success = tracker.export_database_to_csv(csv_path, detailed=True)
                if success:
                    print(f"📂 CSV file ready at: {os.path.abspath(csv_path)}")
            elif key == ord('v'):
                # Export database to CSV (simple format)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = f"exports/terminal_evaluations_simple_{timestamp}.csv"
                success = tracker.export_database_to_csv(csv_path, detailed=False)
                if success:
                    print(f"📂 CSV file ready at: {os.path.abspath(csv_path)}")
            elif key == ord('n'):
                # Export last 10 records to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = f"exports/terminal_evaluations_recent10_{timestamp}.csv"
                success = tracker.export_database_to_csv(csv_path, detailed=True, limit=10)
                if success:
                    print(f"📂 Recent 10 records exported to: {os.path.abspath(csv_path)}")
            elif key == ord('a'):
                # Test Arduino connection
                print("\n🔌 Testing Arduino connection...")
                if tracker.arduino_connected:
                    print("Sending test OK signal...")
                    tracker._send_signal_to_terminal('OK')
                    time.sleep(2)
                    print("Sending test NG signal...")
                    tracker._send_signal_to_terminal('NG')
                    print("✅ Arduino test completed")
                else:
                    print("❌ Arduino not connected - cannot test")
                    print(f"   Port: {tracker.arduino_port}")
                    print("   Check connection and restart program")
            elif key == ord('+') or key == ord('='):
                # Increase result rectangle width
                tracker.result_rectangle_width += 20
                print(f"📏 Result rectangle width increased to {tracker.result_rectangle_width}px (height: {tracker.result_rectangle_height}px)")
            elif key == ord('-') or key == ord('_'):
                # Decrease result rectangle width (minimum 40px)
                tracker.result_rectangle_width = max(40, tracker.result_rectangle_width - 20)
                print(f"📏 Result rectangle width decreased to {tracker.result_rectangle_width}px (height: {tracker.result_rectangle_height}px)")
            elif key == ord('w'):
                # Increase result rectangle height
                tracker.result_rectangle_height += 20
                print(f"📏 Result rectangle height increased to {tracker.result_rectangle_height}px (width: {tracker.result_rectangle_width}px)")
            elif key == ord('x'):
                # Decrease result rectangle height (minimum 40px)
                tracker.result_rectangle_height = max(40, tracker.result_rectangle_height - 20)
                print(f"📏 Result rectangle height decreased to {tracker.result_rectangle_height}px (width: {tracker.result_rectangle_width}px)")
            elif key == ord('z'):
                # Show current result rectangle dimensions
                print(f"📏 Current result rectangle dimensions: {tracker.result_rectangle_width}px × {tracker.result_rectangle_height}px")
            elif key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        # Close Arduino connection
        tracker.close_arduino_connection()
        # Close database connection
        tracker.db_manager.close()
        video_handler.release()
        cv2.destroyAllWindows()
    
    # Enhanced final statistics
    print("\n📊 Final Robust Tracking Statistics")
    print("="*55)
    stats = tracking_results['stats']
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Calculate rejection rates
    total_rejections = (stats['proximity_rejections'] + 
                       stats['depth_rejections'] + 
                       stats['duration_rejections'])
    
    if total_rejections > 0:
        print(f"\n🛡️  False Positive Prevention:")
        print(f"   - Total rejections: {total_rejections}")
        print(f"   - Proximity rejections: {stats['proximity_rejections']} ({100*stats['proximity_rejections']/total_rejections:.1f}%)")
        print(f"   - Depth rejections: {stats['depth_rejections']} ({100*stats['depth_rejections']/total_rejections:.1f}%)")
        print(f"   - Duration rejections: {stats['duration_rejections']} ({100*stats['duration_rejections']/total_rejections:.1f}%)")
    
    print(f"\nRecent Push Events:")
    for i, event in enumerate(tracker.push_events[-5:]):
        print(f"  {i+1}. Connector {event.connector_id} → Terminal {event.terminal_id}")
        print(f"      Push #{event.push_number}, Depth: {event.max_depth:.1f}, Duration: {event.duration_frames} frames")
    
    print("\nTerminal Final States:")
    for i, terminal in enumerate(terminals):
        status = terminal.get('status', 'waiting')
        push_count = tracker.terminal_states.get(i, {}).get('push_count', 0)
        print(f"  Terminal {i+1}: {status} (pushes: {push_count})")
    
    success_rate = 0
    if stats['push_events'] > 0:
        success_rate = (stats['successful_insertions'] / stats['push_events']) * 100
    
    total_db_records = tracker.db_manager.get_record_count()
    print(f"\n✅ Robust tracking test completed")
    print(f"🎯 Success rate: {success_rate:.1f}% ({stats['successful_insertions']}/{stats['push_events']})")
    print(f"💾 Total database records: {total_db_records}")
    print(f"📁 Use CSV export (keys 'c', 'v', 'n') to access stored data")

if __name__ == "__main__":
    print("🎯 Robust Connector Tracking - Arduino Integration")
    print("="*60)
    print("🛡️  Proximity-based push detection to eliminate false positives")
    print("💾 Database integration with on-demand CSV export")
    print("🔌 Arduino integration for OK/NG signal transmission")
    print("   ✅ Only interacts with closest terminal within threshold")
    print("   ✅ Depth smoothing to reduce jitter")
    print("   ✅ Duration validation for sustained pushes")
    print("   ✅ Comprehensive rejection tracking")
    print("   ✅ SQLite database for persistent storage (append mode)")
    print("   ✅ CSV export: Export data when needed")
    print("   ✅ Arduino communication: Sends OK/NG signals to terminal")
    
    # Use hardcoded model path
    model_path = "model/seg-con2.pt"
    print(f"\n📁 Using model: {model_path}")
    
    # Choose video source
    print("\nChoose video source:")
    print("1. Webcam (live testing)")
    print("2. Video file (controlled testing)")
    
    choice = input("Enter choice (1-2): ").strip()
    
    video_source = None
    if choice == "2":
        video_source = "input videos/SampleClip1_2.mp4"
        print(f"📹 Using video: {video_source}")
    elif choice == "1":
        print("📹 Using webcam for live testing")
    else:
        print("Invalid choice. Using webcam as default.")
    
    # Arduino port configuration
    print("\nArduino Configuration:")
    arduino_port = input("Enter Arduino COM port (default: COM3): ").strip()
    if not arduino_port:
        arduino_port = 'COM3'
    
    print(f"🔌 Arduino port: {arduino_port}")
    
    print("\n🚀 Starting robust tracking system with Arduino integration...")
    print("💡 Watch for proximity circles around terminals in debug mode")
    print("💡 Connectors will only interact with terminals they're close to")
    print("💾 Data automatically saved to database (append mode)")
    print("📁 Use keyboard shortcuts to export CSV when needed")
    print("🔌 System will automatically send OK/NG signals to Arduino")
    print("💡 Use 'a' key to test Arduino connection manually")
    
    test_robust_tracking_system(model_path, video_source, arduino_port=arduino_port)