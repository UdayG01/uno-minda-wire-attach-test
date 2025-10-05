import cv2
import numpy as np
import time
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import math
import os

from video_handler import VideoHandler
from terminal_boundary_drawer import TerminalBoundaryDrawer
from connector_detector import ConnectorDetector

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

class RobustConnectorTracker:
    """
    Robust connector tracking with proximity-based push detection
    Eliminates false positives from jittery paths and distant terminals
    """
    
    def __init__(self, max_disappeared: int = 30, 
                 proximity_threshold: float = 80.0,    # Must be within 80px of terminal
                 push_depth_threshold: float = 15.0,   # Depth to consider as push
                 min_push_duration: int = 8,           # Minimum frames for valid push
                 max_push_duration: int = 60,          # Maximum frames for a single push
                 second_push_timeout: int = 150,       # 5 seconds at 30fps
                 continuous_push_timeout: int = 20,    # 2 seconds at 30fps for continuous push success
                 depth_smoothing_window: int = 5,      # Smooth depth over 5 frames
                 status_square_size: int = 120):       # Size of status squares around terminals
        """
        Initialize robust tracking system
        
        Args:
            proximity_threshold: Max distance to assign terminal to connector
            push_depth_threshold: Minimum depth for valid push
            min_push_duration: Minimum frames to sustain push
            max_push_duration: Maximum frames for a single push attempt
            second_push_timeout: Max frames between first and second push
            continuous_push_timeout: Max frames to wait for continuous push before auto-success
            depth_smoothing_window: Frames to smooth depth measurements
            status_square_size: Size of status notification squares around terminals
        """
        self.max_disappeared = max_disappeared
        self.proximity_threshold = proximity_threshold
        self.push_depth_threshold = push_depth_threshold
        self.min_push_duration = min_push_duration
        self.max_push_duration = max_push_duration
        self.second_push_timeout = second_push_timeout
        self.continuous_push_timeout = continuous_push_timeout
        self.depth_smoothing_window = depth_smoothing_window
        self.status_square_size = status_square_size
        
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
        """
        Simple point-in-polygon test using ray casting algorithm
        Returns positive when inside, negative when outside (like cv2.pointPolygonTest)
        """
        if len(terminal_points) < 3:
            return float('-inf')
        
        x, y = point
        inside = False
        n = len(terminal_points)
        
        # Ray casting algorithm
        p1x, p1y = terminal_points[0]
        for i in range(1, n + 1):
            p2x, p2y = terminal_points[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        if inside:
            # Calculate distance to polygon center when inside
            center_x = sum(p[0] for p in terminal_points) / len(terminal_points)
            center_y = sum(p[1] for p in terminal_points) / len(terminal_points)
            distance_to_center = ((x - center_x)**2 + (y - center_y)**2)**0.5
            return distance_to_center  # Positive = inside
        else:
            # Calculate distance to closest edge when outside
            min_dist = float('inf')
            for i in range(len(terminal_points)):
                p1 = terminal_points[i]
                p2 = terminal_points[(i + 1) % len(terminal_points)]
                
                # Distance from point to line segment
                A = x - p1[0]
                B = y - p1[1]
                C = p2[0] - p1[0]
                D = p2[1] - p1[1]
                
                dot = A * C + B * D
                len_sq = C * C + D * D
                
                if len_sq == 0:
                    # Point is same as line start
                    dist = math.sqrt(A * A + B * B)
                else:
                    param = dot / len_sq
                    
                    if param < 0:
                        # Closest point is p1
                        xx = p1[0]
                        yy = p1[1]
                    elif param > 1:
                        # Closest point is p2
                        xx = p2[0]
                        yy = p2[1]
                    else:
                        # Closest point is on the line segment
                        xx = p1[0] + param * C
                        yy = p1[1] + param * D
                    
                    dx = x - xx
                    dy = y - yy
                    dist = math.sqrt(dx * dx + dy * dy)
                
                min_dist = min(min_dist, dist)
            
            return -min_dist  # Negative = outside
    
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
        
        # Step 2: Check if closest terminal is locked
        if self._is_terminal_locked(closest_terminal):
            # Terminal is locked - don't assign or process
            if track.assigned_terminal == closest_terminal:
                # Clear assignment to locked terminal
                track.assigned_terminal = None
                track.current_session = None
            return
        
        # Step 3: Assign terminal if not already assigned or if closer terminal found
        if track.assigned_terminal != closest_terminal:
            track.assigned_terminal = closest_terminal
            track.current_session = None  # Reset session when switching terminals
        
        # Step 4: Calculate depth to assigned terminal only
        terminal_points = terminals[track.assigned_terminal]['points']
        raw_depth = self._point_to_terminal_distance(track.current_centroid, terminal_points)
        smoothed_depth = self._smooth_depth_measurement(track, raw_depth)
        
        # Step 5: Analyze push state for assigned terminal (will check lock again inside)
        self._update_push_session(track, track.assigned_terminal, smoothed_depth, terminals)
    
    def _update_push_session(self, track: TrackedConnector, terminal_id: int, depth: float, terminals: List[Dict[str, Any]]):
        """
        Update push session for connector-terminal pair
        """
        # CHECK LOCK IMMEDIATELY - don't process any push for locked terminals
        if self._is_terminal_locked(terminal_id):
            # Clear any active session for this locked terminal
            if track.current_session and track.assigned_terminal == terminal_id:
                track.current_session = None
                track.assigned_terminal = None
            return
        
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
                'last_push_frame': None,
                'first_push_completed_frame': None  # Track when first push was completed
            }
        
        session = track.current_session
        terminal_state = self.terminal_states[terminal_id]
        
        # State machine for robust push detection
        if session['state'] == 'outside':
            if depth > 5:  # Entered terminal
                session['state'] = 'inside'
                session['current_push_depth'] = depth
                session['max_depth'] = depth
                print(f"🔄 Connector {track.id} entered terminal {terminal_id} (depth: {depth:.1f})")
        
        elif session['state'] == 'inside':
            if depth > 5:  # Still inside
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
                session['first_push_completed_frame'] = None
                self.tracking_stats['depth_rejections'] += 1
        
        elif session['state'] == 'pushing':
            if depth > 0:  # Still inside
                session['current_push_depth'] = depth
                session['max_depth'] = max(session['max_depth'], depth)
                
                # Calculate current push duration
                current_push_duration = self.frame_count - session['push_start_frame']
                
                # CHECK FOR MAXIMUM PUSH DURATION FIRST
                if current_push_duration >= self.max_push_duration:
                    # Push has gone on too long - force completion
                    print(f"⏰ FORCE ENDING: Connector {track.id} push exceeded max duration ({current_push_duration} frames)")
                    completed = current_push_duration >= self.min_push_duration  # Should be True
                    self._end_push_attempt(track, terminal_id, session, terminals, completed)
                    return
                
                # Still above threshold - continue pushing
                if depth >= self.push_depth_threshold:
                    # Check for continuous push timeout after first push is completed
                    if (session.get('first_push_completed_frame') is not None and 
                        terminal_state['push_count'] == 1):
                        frames_since_first_push = self.frame_count - session['first_push_completed_frame']
                        
                        if frames_since_first_push >= self.continuous_push_timeout:
                            # Continuous push for 2+ seconds after first push - treat as success
                            if not self._is_terminal_locked(terminal_id):
                                terminals[terminal_id]['color'] = (0, 255, 0)  # Green
                                terminals[terminal_id]['status'] = 'success'
                                terminal_state['push_count'] = 2
                                self._lock_terminal_status(terminal_id)
                                
                                self.tracking_stats['successful_insertions'] += 1
                                print(f"✅ CONTINUOUS PUSH SUCCESS: Connector {track.id} → Terminal {terminal_id} (2+ seconds) - STATUS LOCKED")
                                
                                # Reset session state
                                session['state'] = 'outside'
                                session['push_start_frame'] = None
                                session['current_push_depth'] = 0.0
                                session['max_depth'] = 0.0
                                return  # Exit to prevent further processing
                    
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
        # CHECK LOCK FIRST - if terminal is locked, reject any new pushes
        if self._is_terminal_locked(terminal_id):
            print(f"🔒 Terminal {terminal_id} is locked - ignoring push from connector {track.id}")
            session['state'] = 'outside'
            session['push_start_frame'] = None
            session['current_push_depth'] = 0.0
            session['max_depth'] = 0.0
            session['first_push_completed_frame'] = None
            return
        
        # Store push_start_frame before clearing it
        push_start_frame = session['push_start_frame']
        
        if not completed:
            # Push too short - reject
            session['state'] = 'outside'
            session['push_start_frame'] = None
            session['current_push_depth'] = 0.0
            session['max_depth'] = 0.0
            session['first_push_completed_frame'] = None
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
        
        # Update terminal state based on push count - BUT CHECK LOCK AGAIN
        if self._is_terminal_locked(terminal_id):
            print(f"🔒 Terminal {terminal_id} became locked during processing - ignoring status change")
            return
        
        # Update terminal state based on push count
        if session['push_count'] == 1:
            # First push - only change if not locked
            terminals[terminal_id]['color'] = (0, 255, 255)  # Yellow
            terminals[terminal_id]['status'] = 'first_push'
            terminal_state['push_count'] = 1
            terminal_state['last_push_frame'] = self.frame_count
            
            # Record when first push was completed for continuous push timeout tracking
            session['first_push_completed_frame'] = self.frame_count
            
            print(f"🎯 FIRST PUSH: Connector {track.id} → Terminal {terminal_id} (Max depth: {session['max_depth']:.1f}px)")

            
        
        elif session['push_count'] == 2:
            # Check timing for second push
            if terminal_state['last_push_frame']:
                frames_since_first = self.frame_count - terminal_state['last_push_frame']
                
                if frames_since_first <= self.second_push_timeout:
                    # Valid second push - check lock one more time before success
                    if not self._is_terminal_locked(terminal_id):
                        terminals[terminal_id]['color'] = (0, 255, 0)  # Green
                        terminals[terminal_id]['status'] = 'success'
                        terminal_state['push_count'] = 2
                        self._lock_terminal_status(terminal_id)
                        
                        self.tracking_stats['successful_insertions'] += 1
                        print(f"✅ SUCCESS: Connector {track.id} → Terminal {terminal_id} (Second push) - STATUS LOCKED")
                    else:
                        print(f"🔒 Terminal {terminal_id} status already locked - ignoring second push")
                else:
                    # Timeout - treat as first push of new sequence (only if not locked)
                    if not self._is_terminal_locked(terminal_id):
                        terminals[terminal_id]['color'] = (0, 255, 255)  # Yellow
                        terminals[terminal_id]['status'] = 'first_push'
                        terminal_state['push_count'] = 1
                        terminal_state['last_push_frame'] = self.frame_count
                        session['push_count'] = 1  # Reset session count
                        session['first_push_completed_frame'] = None  # Reset continuous push tracking
                        
                        print(f"⏰ TIMEOUT: Terminal {terminal_id} reset, treating as first push")
                    else:
                        print(f"🔒 Terminal {terminal_id} is locked - ignoring timeout reset")
            else:
                # No previous push recorded - treat as first (only if not locked)
                if not self._is_terminal_locked(terminal_id):
                    terminals[terminal_id]['color'] = (0, 255, 255)  # Yellow
                    terminals[terminal_id]['status'] = 'first_push'
                    terminal_state['push_count'] = 1
                    terminal_state['last_push_frame'] = self.frame_count
                    session['push_count'] = 1
                    session['first_push_completed_frame'] = None  # Reset continuous push tracking
        
        # Reset depth tracking and clear push_start_frame AFTER using it
        session['max_depth'] = 0.0
        session['push_start_frame'] = None
    
    def _check_for_timeouts(self, terminals: List[Dict[str, Any]]):
        """Check for terminals waiting too long for second push"""
        for terminal_id, terminal_state in self.terminal_states.items():
            # ALWAYS CHECK LOCK FIRST
            if self._is_terminal_locked(terminal_id):
                continue  # Skip locked terminals completely
            
            if (terminal_state['push_count'] == 1 and 
                terminal_state['last_push_frame'] is not None):
                
                frames_since_push = self.frame_count - terminal_state['last_push_frame']
                
                if frames_since_push > self.second_push_timeout:
                    # Timeout - only change status if not locked (double check)
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
        Send OK/NG signal to terminal
        Replace this with actual implementation (serial, TCP, etc.)
        """
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"📡 [{timestamp}] SENDING SIGNAL TO TERMINAL: {signal}")
        
        # TODO: Replace with actual terminal communication
        # Examples:
        # serial_port.write(signal.encode())
        # tcp_socket.send(signal.encode())
        # subprocess.run(['echo', signal, '>', '/dev/ttyUSB0'])
        
        print(f"🔗 Signal '{signal}' sent successfully")
    
    def _reset_all_terminals(self, terminals: List[Dict[str, Any]]):
        """Reset all terminals to orange (waiting) status"""
        print("\n🔄 RESETTING ALL TERMINALS...")
        
        for i, terminal in enumerate(terminals):
            terminal['color'] = (0, 165, 255)  # Orange
            terminal['status'] = 'waiting'
        
        # Clear all terminal states and locks
        self.terminal_states.clear()
        self.terminal_status_locked.clear()
        
        # Reset tracking statistics for new evaluation cycle
        self.tracking_stats.update({
            'push_events': 0,
            'successful_insertions': 0,
            'proximity_rejections': 0,
            'depth_rejections': 0,
            'duration_rejections': 0
        })
        
        print("✅ All terminals reset to ORANGE (waiting) status")
        print("🔓 All terminal locks cleared")
        print("📊 Statistics reset for new evaluation cycle")
        print("="*60)
    
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
                cv2.circle(display_frame, center, 2, (255, 255, 255), -1)
                
                # Draw proximity threshold circle in debug mode
                if debug_mode:
                    cv2.circle(display_frame, center, int(self.proximity_threshold), (128, 128, 128), 1)
                
                # Draw status notification square for SUCCESS or FAILED terminals
                status = terminal.get('status', 'waiting')
                if status in ['success', 'failed']:
                    self._draw_status_square(display_frame, center, status, i+1)
            
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
        
        # Terminal Status Table (always visible in top right)
        self._draw_terminal_status_table(display_frame, terminals)
        
        # Enhanced statistics overlay (below status table when debug mode is on)
        if debug_mode:
            stats = tracking_results['stats']
            overlay_height = 180  # Reduced height to accommodate status table
            overlay = np.zeros((overlay_height, 400, 3), dtype=np.uint8)
            
            cv2.putText(overlay, "ROBUST TRACKING STATS", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay, f"Active tracks: {stats['active_tracks']}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(overlay, f"Push events: {stats['push_events']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(overlay, f"Successful: {stats['successful_insertions']}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(overlay, f"Proximity rejections: {stats['proximity_rejections']}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(overlay, f"Depth rejections: {stats['depth_rejections']}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(overlay, f"Duration rejections: {stats['duration_rejections']}", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(overlay, f"Evaluations: {len(self.evaluation_results)}", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 255, 128), 1)
            
            # Place overlay below status table (TOP RIGHT corner)
            frame_width = display_frame.shape[1]
            overlay_width = 400
            x_offset = frame_width - overlay_width - 10  # 10px margin from right edge
            y_offset = 280  # Increased from 200 to accommodate larger status table
            display_frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = overlay
        
        return display_frame
    
    def _draw_status_square(self, frame: np.ndarray, center: Tuple[int, int], status: str, terminal_id: int):
        """
        Draw a colored square around terminal with status message and terminal ID
        
        Args:
            frame: The frame to draw on
            center: Center point of the terminal
            status: Terminal status ('success' or 'failed')
            terminal_id: Terminal ID number (1, 2, 3, etc.)
        """
        half_size = self.status_square_size // 2
        
        # Calculate square corners - properly centered around the terminal
        x1 = center[0] - half_size
        y1 = center[1] - half_size
        x2 = center[0] + half_size
        y2 = center[1] + half_size
        
        # Ensure square is within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width - 1, x2)
        y2 = min(frame_height - 1, y2)
        
        # Determine color and message based on status
        if status == 'success':
            square_color = (0, 255, 0)  # Green
            message = "OK"
            text_color = (255, 255, 255)  # White text
        elif status == 'failed':
            square_color = (0, 0, 255)  # Red
            message = "NG"
            text_color = (255, 255, 255)  # White text
        else:
            return  # Don't draw for other statuses
        
        # Draw thick square outline
        square_thickness = 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), square_color, square_thickness)
        
        # Create semi-transparent background overlay for the entire square
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), square_color, -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Terminal ID - larger text at the top of square
        terminal_text = f"T{terminal_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_id = 1.2
        font_thickness_id = 3
        
        # Get terminal ID text dimensions
        (id_text_width, id_text_height), _ = cv2.getTextSize(terminal_text, font, font_scale_id, font_thickness_id)
        
        # Position terminal ID in upper part of square
        id_text_x = center[0] - id_text_width // 2
        id_text_y = center[1] - 10  # Slightly above center
        
        # Draw terminal ID with outline for better visibility
        cv2.putText(frame, terminal_text, (id_text_x, id_text_y), font, font_scale_id, (0, 0, 0), font_thickness_id + 2)  # Black outline
        cv2.putText(frame, terminal_text, (id_text_x, id_text_y), font, font_scale_id, text_color, font_thickness_id)  # White text
        
        # Status message - smaller text at the bottom of square
        font_scale_status = 0.8
        font_thickness_status = 2
        
        # Get status text dimensions
        (status_text_width, status_text_height), _ = cv2.getTextSize(message, font, font_scale_status, font_thickness_status)
        
        # Position status message in lower part of square
        status_text_x = center[0] - status_text_width // 2
        status_text_y = center[1] + 25  # Below center
        
        # Draw status text with outline for better visibility
        cv2.putText(frame, message, (status_text_x, status_text_y), font, font_scale_status, (0, 0, 0), font_thickness_status + 2)  # Black outline
        cv2.putText(frame, message, (status_text_x, status_text_y), font, font_scale_status, text_color, font_thickness_status)  # White text
    
    def _draw_terminal_status_table(self, frame: np.ndarray, terminals: List[Dict[str, Any]]):
        """
        Draw terminal status table in the top right corner
        """
        # Table dimensions and positioning 
        frame_height, frame_width = frame.shape[:2]
        table_width = 350  # from 250
        header_height = 35  # from 25
        row_height = 30     # from 20
        num_terminals = len(terminals)
        overall_status_height = 35  # from 25
        table_height = header_height + (num_terminals * row_height) + overall_status_height + 30  # +30 for padding
        
        # Position in top right corner
        x_offset = frame_width - table_width - 10
        y_offset = 10
        
        # Create table background
        table_overlay = np.zeros((table_height, table_width, 3), dtype=np.uint8)
        table_overlay.fill(40)  # Dark gray background
        
        # Draw border
        cv2.rectangle(table_overlay, (0, 0), (table_width-1, table_height-1), (255, 255, 255), 2)
        
        # Header
        cv2.putText(table_overlay, "TERMINAL STATUS", (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Larger font
        
        # Draw header separator line
        cv2.line(table_overlay, (10, header_height), (table_width-10, header_height), (255, 255, 255), 2)  # Thicker line
        
        # Terminal status rows
        all_ok = True
        has_waiting = False
        has_first_push = False
        has_failed = False
        
        for i, terminal in enumerate(terminals):
            y_pos = header_height + 10 + (i * row_height) + 20  # Adjusted spacing
            
            # Get terminal status
            status = terminal.get('status', 'waiting')
            color = terminal.get('color', (0, 165, 255))  # Default orange
            is_locked = self._is_terminal_locked(i)
            
            # Track status types for overall status calculation
            if status == 'success':
                pass  # Keep all_ok = True for this terminal
            elif status == 'waiting':
                has_waiting = True
                all_ok = False
            elif status == 'first_push':
                has_first_push = True
                all_ok = False
            elif status == 'failed':
                has_failed = True
                all_ok = False
            else:
                all_ok = False
            
            # Determine status text and color for display
            status_text = "NG"
            status_color = (0, 0, 255)  # Red
            
            if status == 'success':
                status_text = "OK"
                status_color = (0, 255, 0)  # Green
            elif status == 'waiting':
                status_text = "WAIT"
                status_color = (0, 165, 255)  # Orange
            elif status == 'first_push':
                status_text = "PUSH1"
                status_color = (0, 255, 255)  # Yellow
            elif status == 'failed':
                status_text = "NG"
                status_color = (0, 0, 255)  # Red
            
            # Draw terminal number (larger font)
            cv2.putText(table_overlay, f"T{i+1}:", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Larger font
            
            # Draw status indicator circle (larger)
            circle_x = 90  # Adjusted position
            circle_radius = 8  # Increased from 6
            cv2.circle(table_overlay, (circle_x, y_pos-7), circle_radius, status_color, -1)
            cv2.circle(table_overlay, (circle_x, y_pos-7), circle_radius, (255, 255, 255), 2)  # Thicker border
            
            # Draw status text (larger font)
            cv2.putText(table_overlay, status_text, (120, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)  # Larger font
            
            # Draw lock indicator if locked (larger font)
            if is_locked:
                cv2.putText(table_overlay, "LOCK", (220, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Overall status section
        overall_y = header_height + 15 + (num_terminals * row_height) + 15  # Adjusted spacing
        
        # Draw separator line before overall status (thicker)
        cv2.line(table_overlay, (10, overall_y - 10), (table_width-10, overall_y - 10), (255, 255, 255), 2)
        
        # Overall status logic - priority: waiting > first_push > failed > OK
        if all_ok:
            overall_status = "OK"
            overall_color = (0, 255, 0)  # Green
        elif has_waiting:
            overall_status = "WAIT"
            overall_color = (0, 165, 255)  # Orange
        elif has_first_push:
            overall_status = "PUSH1"
            overall_color = (0, 255, 255)  # Yellow
        elif has_failed:
            overall_status = "NG"
            overall_color = (0, 0, 255)  # Red
        else:
            overall_status = "NG"
            overall_color = (0, 0, 255)  # Red
        
        cv2.putText(table_overlay, "OVERALL:", (15, overall_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Larger font
        
        # Overall status indicator (larger)
        circle_x = 160  # Adjusted position
        cv2.circle(table_overlay, (circle_x, overall_y + 12), 10, overall_color, -1)  # Larger circle
        cv2.circle(table_overlay, (circle_x, overall_y + 12), 10, (255, 255, 255), 2)  # Thicker border
        
        cv2.putText(table_overlay, overall_status, (185, overall_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, overall_color, 2)  # Larger font
        
        # Apply table to frame with some transparency
        roi = frame[y_offset:y_offset+table_height, x_offset:x_offset+table_width]
        if roi.shape[0] == table_overlay.shape[0] and roi.shape[1] == table_overlay.shape[1]:
            # Blend with alpha
            alpha = 0.9
            frame[y_offset:y_offset+table_height, x_offset:x_offset+table_width] = \
                cv2.addWeighted(roi, 1-alpha, table_overlay, alpha, 0)

def test_robust_tracking_system(model_path: str, video_source: Optional[str] = None,
                               config_file: str = "terminal_config.json",
                               output_video_path: str = "output videos/push_detector2.mp4"):
    """
    Test the robust proximity-based tracking system
    """
    print("🎯 ROBUST CONNECTOR TRACKING & PUSH DETECTION TEST")
    print("="*65)
    print("🛡️  Proximity-based push detection enabled")
    print("   - Proximity threshold: 80.0 pixels (only closest terminal)")
    print("   - Push depth threshold: 15.0 pixels")
    print("   - Minimum push duration: 8 frames (~0.27s)")
    print("   - Second push timeout: 150 frames (5.0s)")
    print("   - Depth smoothing: 5 frame median filter")
    
    # Initialize components
    print("\n🚀 Initializing components...")
    
    # Video handler
    video_handler = VideoHandler(video_source, target_fps=30)
    if not video_handler.initialize():
        print("❌ Failed to initialize video")
        return

        # Print actual resolution being used
    if video_handler.is_webcam:
        width = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📐 Using webcam resolution: {width}x{height}")
    
    # Terminal boundary drawer
    boundary_drawer = TerminalBoundaryDrawer(config_file=config_file)
    if not boundary_drawer.load_configuration():
        print("❌ Failed to load terminal configuration")
        return
    
    terminals = boundary_drawer.terminals
    print(f"✅ Loaded {len(terminals)} terminals")
    
    # Connector detector
    detector = ConnectorDetector(model_path, confidence_threshold=0.25, target_classes=['connector'])
    if not detector.initialize_model():
        print("❌ Failed to initialize detector")
        return
    
    # Robust connector tracker
    tracker = RobustConnectorTracker(
        max_disappeared=10,
        proximity_threshold=40.0,        # Must be within 40px of terminal center
        push_depth_threshold=0.25,       # 1px inside terminal boundary
        min_push_duration=3,             # ~0.27 seconds minimum push
        max_push_duration=15,            # ~3 seconds maximum push (90 frames at 30fps)
        second_push_timeout=25,         # 5 seconds for second push
        depth_smoothing_window=5,        # Smooth depth over 5 frames
        status_square_size=150           # Size of status notification squares
    )
    
    print("✅ Robust tracking system initialized")
    
    # Setup video writer
    video_writer = None
    if output_video_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created output directory: {output_dir}")
        
        # Get video properties for output
        frame_width = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_handler.target_fps
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        if video_writer.isOpened():
            print(f"📹 Video writer initialized: {output_video_path}")
            print(f"   Resolution: {frame_width}x{frame_height} @ {fps}fps")
        else:
            print(f"❌ Failed to initialize video writer")
            video_writer = None
    
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
    print("  'r': Reset terminal states")
    print("  't': Print tracking thresholds")
    print("  's': Print current statistics")
    print("  'e': Show detailed push event history")
    print("  'q': Quit and save video")
    
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
            
            # Recording indicator
            if video_writer and video_writer.isOpened():
                cv2.circle(display_frame, (display_frame.shape[1] - 30, 50), 8, (0, 0, 255), -1)
                cv2.putText(display_frame, "REC", 
                           (display_frame.shape[1] - 55, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Mode indicators
            if debug_mode:
                cv2.putText(display_frame, "DEBUG: Proximity analysis ON", 
                           (display_frame.shape[1] - 250, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if paused:
                cv2.putText(display_frame, "PAUSED", 
                           (display_frame.shape[1]//2 - 50, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Write frame to output video
            if video_writer and video_writer.isOpened() and not paused:
                video_writer.write(display_frame)
            
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
                if paused:
                    print("   📹 Video recording paused")
                else:
                    print("   📹 Video recording resumed")
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
            elif key == ord('s'):
                stats = tracking_results['stats']
                print("\n📊 Current Tracking Statistics:")
                print(f"   - Active tracks: {stats['active_tracks']}")
                print(f"   - Total tracks created: {stats['total_tracks']}")
                print(f"   - Push events: {stats['push_events']}")
                print(f"   - Successful insertions: {stats['successful_insertions']}")
                print(f"   - Proximity rejections: {stats['proximity_rejections']}")
                print(f"   - Depth rejections: {stats['depth_rejections']}")
                print(f"   - Duration rejections: {stats['duration_rejections']}")
                
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
            elif key == ord('h'):
                # Show evaluation history
                print("\n📋 Evaluation History:")
                if tracker.evaluation_results:
                    for i, eval_result in enumerate(tracker.evaluation_results):
                        timestamp_str = time.strftime("%H:%M:%S", time.localtime(eval_result['timestamp']))
                        result = eval_result['result']
                        success_count = eval_result['successful_terminals']
                        total_count = eval_result['total_terminals']
                        print(f"   Evaluation #{i+1}: {timestamp_str} - {result} ({success_count}/{total_count} terminals)")
                else:
                    print("   No evaluations performed yet")
            elif key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        # Release video writer
        if video_writer and video_writer.isOpened():
            video_writer.release()
            print(f"💾 Video saved to: {output_video_path}")
            
            # Check if file was created successfully
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                print(f"   File size: {file_size / (1024*1024):.2f} MB")
                print(f"   Frames processed: {frame_count}")
            else:
                print(f"❌ Failed to save video file")
        
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
    
    print(f"\n✅ Robust tracking test completed")
    print(f"🎯 Success rate: {success_rate:.1f}% ({stats['successful_insertions']}/{stats['push_events']})")

if __name__ == "__main__":
    print("🎯 Robust Connector Tracking - Step 6 Final")
    print("="*50)
    print("🛡️  Proximity-based push detection to eliminate false positives")
    print("   ✅ Only interacts with closest terminal within threshold")
    print("   ✅ Depth smoothing to reduce jitter")
    print("   ✅ Duration validation for sustained pushes")
    print("   ✅ Comprehensive rejection tracking")
    
    # Use hardcoded model path
    model_path = "model/seg-con2.pt"
    print(f"\n📁 Using model: {model_path}")
    
    # Output video path
    OUTPUT_VIDEO_PATH = "output videos/push_detector2.mp4"
    print(f"📹 Output video will be saved to: {OUTPUT_VIDEO_PATH}")
    
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
    
    print("\n🚀 Starting robust tracking system...")
    print("💡 Watch for proximity circles around terminals in debug mode")
    print("💡 Connectors will only interact with terminals they're close to")
    print("🔴 Recording indicator will show in top-right corner")
    
    test_robust_tracking_system(model_path, video_source, output_video_path=OUTPUT_VIDEO_PATH)