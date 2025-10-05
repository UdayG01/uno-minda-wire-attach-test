import psycopg2
import psycopg2.pool
import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PostgresDatabaseManager:
    """
    PostgreSQL database manager for storing terminal status data with CSV export
    """
    
    def __init__(self):
        """Initialize PostgreSQL database manager"""
        self.connection_pool = None
        self.config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'pushdetection'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password'),
            'minconn': int(os.getenv('POSTGRES_POOL_MIN', 1)),
            'maxconn': int(os.getenv('POSTGRES_POOL_MAX', 10))
        }
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection pool and create tables"""
        try:
            # Create connection pool
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config['minconn'],
                maxconn=self.config['maxconn'],
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            
            # Test connection and create tables
            connection = self.connection_pool.getconn()
            try:
                cursor = connection.cursor()
                
                # Create evaluations table with enhanced schema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evaluations (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        overall_status VARCHAR(10) NOT NULL CHECK (overall_status IN ('OK', 'NG', 'WAIT')),
                        terminal_statuses JSONB NOT NULL,
                        terminal_count INTEGER NOT NULL,
                        successful_terminals INTEGER NOT NULL DEFAULT 0,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp 
                    ON evaluations (timestamp DESC)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_evaluations_status 
                    ON evaluations (overall_status)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_evaluations_terminal_data 
                    ON evaluations USING GIN (terminal_statuses)
                """)
                
                connection.commit()
                print(f"✅ PostgreSQL database initialized successfully")
                print(f"   🏠 Host: {self.config['host']}:{self.config['port']}")
                print(f"   🗄️  Database: {self.config['database']}")
                print(f"   👤 User: {self.config['user']}")
                
            finally:
                cursor.close()
                self.connection_pool.putconn(connection)
                
        except psycopg2.Error as e:
            print(f"❌ PostgreSQL initialization error: {e}")
            self.connection_pool = None
        except Exception as e:
            print(f"❌ Database setup error: {e}")
            self.connection_pool = None
    
    def save_evaluation(self, terminals: List[Dict[str, Any]]) -> Optional[int]:
        """
        Save evaluation data to PostgreSQL database
        
        Args:
            terminals: List of terminal configurations with current status
            
        Returns:
            Evaluation ID if successful, None if failed
        """
        if not self.connection_pool:
            print("❌ Database pool not available")
            return None
        
        connection = None
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor()
            
            # Determine overall status
            has_wait = False
            has_fail = False
            successful_count = 0
            
            # Parse terminal data into structured format
            terminal_data = []
            for i, terminal in enumerate(terminals):
                color = terminal.get('color', (0, 165, 255))
                status = terminal.get('status', 'waiting')
                
                # Convert BGR color to status
                if color == (0, 255, 0):  # Green
                    simple_status = "OK"
                    successful_count += 1
                elif color == (0, 0, 255):  # Red
                    simple_status = "NG"
                    has_fail = True
                else:  # Orange or Yellow
                    simple_status = "WAIT"
                    has_wait = True
                
                terminal_info = {
                    'terminal_id': i + 1,
                    'status': simple_status,
                    'color_bgr': color,
                    'raw_status': status
                }
                terminal_data.append(terminal_info)
            
            # Determine overall status: WAIT > NG > OK
            if has_wait:
                overall_status = "WAIT"
            elif has_fail:
                overall_status = "NG"
            else:
                overall_status = "OK"
            
            # Insert evaluation record
            cursor.execute("""
                INSERT INTO evaluations 
                (overall_status, terminal_statuses, terminal_count, successful_terminals, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, timestamp
            """, (
                overall_status,
                json.dumps(terminal_data),
                len(terminals),
                successful_count,
                json.dumps({
                    'version': '1.0',
                    'source': 'push_detector9',
                    'total_terminals': len(terminals)
                })
            ))
            
            result = cursor.fetchone()
            evaluation_id, timestamp = result
            
            connection.commit()
            
            # Create summary string for logging
            terminal_summary = ", ".join([
                f"T{t['terminal_id']}:{t['status']}" for t in terminal_data
            ])
            
            print(f"✅ Evaluation saved to PostgreSQL (ID: {evaluation_id}):")
            print(f"   📅 Timestamp: {timestamp}")
            print(f"   🎯 Overall Status: {overall_status}")
            print(f"   📋 Terminals: {terminal_summary}")
            print(f"   ✅ Success Rate: {successful_count}/{len(terminals)}")
            
            return evaluation_id
            
        except psycopg2.Error as e:
            print(f"❌ PostgreSQL save error: {e}")
            if connection:
                connection.rollback()
            return None
        except Exception as e:
            print(f"❌ Database save error: {e}")
            if connection:
                connection.rollback()
            return None
        finally:
            if connection:
                cursor.close()
                self.connection_pool.putconn(connection)
    
    def export_to_csv_detailed(self, csv_path: str, limit: Optional[int] = None) -> bool:
        """
        Export evaluation data to CSV with individual terminal columns
        
        Args:
            csv_path: Path where CSV file will be saved
            limit: Optional limit on number of records (None = all records)
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.connection_pool:
            print("❌ Database pool not available for export")
            return False
        
        connection = None
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor()
            
            # Query data
            if limit:
                cursor.execute("""
                    SELECT id, timestamp, overall_status, terminal_statuses, 
                           terminal_count, successful_terminals
                    FROM evaluations 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id, timestamp, overall_status, terminal_statuses, 
                           terminal_count, successful_terminals
                    FROM evaluations 
                    ORDER BY timestamp ASC
                """)
            
            rows = cursor.fetchall()
            
            if not rows:
                print("❌ No data found to export")
                return False
            
            # Determine max number of terminals
            max_terminals = max(row[4] for row in rows)  # terminal_count column
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
            
            # Write to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header with individual terminal columns
                header = ['ID', 'Timestamp', 'Overall_Status', 'Successful_Terminals', 'Total_Terminals']
                for i in range(max_terminals):
                    header.append(f'Terminal_{i+1}')
                writer.writerow(header)
                
                # Write data rows
                for row in rows:
                    evaluation_id, timestamp, overall_status, terminal_statuses, terminal_count, successful_terminals = row
                    
                    csv_row = [
                        evaluation_id,
                        timestamp.isoformat() if timestamp else '',
                        overall_status,
                        successful_terminals,
                        terminal_count
                    ]
                    
                    # Parse terminal data from JSONB
                    terminal_data = []
                    if terminal_statuses:
                        # Handle both string and already parsed JSON
                        if isinstance(terminal_statuses, str):
                            terminal_data = json.loads(terminal_statuses)
                        elif isinstance(terminal_statuses, list):
                            terminal_data = terminal_statuses
                        else:
                            # If it's already a Python object (dict/list), use as is
                            terminal_data = terminal_statuses
                    
                    # Create terminal dictionary for easy lookup
                    terminal_dict = {}
                    if isinstance(terminal_data, list):
                        for t in terminal_data:
                            if isinstance(t, dict) and 'terminal_id' in t and 'status' in t:
                                terminal_dict[f"T{t['terminal_id']}"] = t['status']
                    
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
        finally:
            if connection:
                cursor.close()
                self.connection_pool.putconn(connection)
    
    def export_to_csv_simple(self, csv_path: str, limit: Optional[int] = None) -> bool:
        """
        Export evaluation data to simple CSV format
        
        Args:
            csv_path: Path where CSV file will be saved
            limit: Optional limit on number of records (None = all records)
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.connection_pool:
            print("❌ Database pool not available for export")
            return False
        
        connection = None
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor()
            
            # Query data
            if limit:
                cursor.execute("""
                    SELECT id, timestamp, overall_status, terminal_statuses, 
                           terminal_count, successful_terminals
                    FROM evaluations 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id, timestamp, overall_status, terminal_statuses, 
                           terminal_count, successful_terminals
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
                writer.writerow(['ID', 'Timestamp', 'Overall_Status', 'Successful_Terminals', 'Total_Terminals', 'Terminal_Summary'])
                
                # Write data rows
                for row in rows:
                    evaluation_id, timestamp, overall_status, terminal_statuses, terminal_count, successful_terminals = row
                    
                    # Create terminal summary string
                    terminal_summary = ""
                    if terminal_statuses:
                        # Handle both string and already parsed JSON
                        terminal_data = []
                        if isinstance(terminal_statuses, str):
                            terminal_data = json.loads(terminal_statuses)
                        elif isinstance(terminal_statuses, list):
                            terminal_data = terminal_statuses
                        else:
                            terminal_data = terminal_statuses
                        
                        if isinstance(terminal_data, list):
                            terminal_summary = ",".join([
                                f"T{t['terminal_id']}:{t['status']}" 
                                for t in terminal_data 
                                if isinstance(t, dict) and 'terminal_id' in t and 'status' in t
                            ])
                    
                    csv_row = [
                        evaluation_id,
                        timestamp.isoformat() if timestamp else '',
                        overall_status,
                        successful_terminals,
                        terminal_count,
                        terminal_summary
                    ]
                    
                    writer.writerow(csv_row)
            
            print(f"✅ Successfully exported {len(rows)} records to simple CSV: {csv_path}")
            print(f"   📊 File size: {os.path.getsize(csv_path)} bytes")
            
            return True
            
        except Exception as e:
            print(f"❌ Simple CSV export error: {e}")
            return False
        finally:
            if connection:
                cursor.close()
                self.connection_pool.putconn(connection)
    
    def get_record_count(self) -> int:
        """Get total number of records in database"""
        if not self.connection_pool:
            return 0
        
        connection = None
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            count = cursor.fetchone()[0]
            return count
        except psycopg2.Error:
            return 0
        finally:
            if connection:
                cursor.close()
                self.connection_pool.putconn(connection)
    
    def get_recent_evaluations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent evaluation records
        
        Args:
            limit: Maximum number of evaluations to retrieve
            
        Returns:
            List of evaluation dictionaries
        """
        if not self.connection_pool:
            return []
        
        connection = None
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("""
                SELECT id, timestamp, overall_status, terminal_statuses, 
                       terminal_count, successful_terminals
                FROM evaluations 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (limit,))
            
            evaluations = []
            for row in cursor.fetchall():
                # Parse terminal data for summary
                terminal_summary = ""
                if row[3]:  # terminal_statuses
                    try:
                        terminal_data = row[3] if isinstance(row[3], list) else json.loads(row[3]) if isinstance(row[3], str) else row[3]
                        if isinstance(terminal_data, list):
                            terminal_summary = ",".join([
                                f"T{t['terminal_id']}:{t['status']}" 
                                for t in terminal_data 
                                if isinstance(t, dict) and 'terminal_id' in t and 'status' in t
                            ])
                    except:
                        terminal_summary = "Parse Error"
                
                evaluations.append({
                    'id': row[0],
                    'timestamp': row[1].isoformat() if row[1] else '',
                    'overall_status': row[2],
                    'terminal_statuses': terminal_summary,
                    'terminal_count': row[4],
                    'successful_terminals': row[5]
                })
            
            return evaluations
            
        except psycopg2.Error as e:
            print(f"❌ Database query error: {e}")
            return []
        finally:
            if connection:
                cursor.close()
                self.connection_pool.putconn(connection)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all evaluations
        
        Returns:
            Dictionary with statistical data
        """
        if not self.connection_pool:
            return {}
        
        connection = None
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor()
            
            # Overall evaluation stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    SUM(CASE WHEN overall_status = 'OK' THEN 1 ELSE 0 END) as ok_evaluations,
                    SUM(CASE WHEN overall_status = 'NG' THEN 1 ELSE 0 END) as ng_evaluations,
                    SUM(CASE WHEN overall_status = 'WAIT' THEN 1 ELSE 0 END) as wait_evaluations,
                    AVG(successful_terminals::float / terminal_count::float) as avg_success_rate
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
                'avg_success_rate': float(stats_row[4] or 0),
                'latest_evaluation': latest_evaluation
            }
            
        except psycopg2.Error as e:
            print(f"❌ Database statistics error: {e}")
            return {}
        finally:
            if connection:
                cursor.close()
                self.connection_pool.putconn(connection)
    
    def close(self):
        """Close database connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None
            print("🔒 PostgreSQL connection pool closed")
