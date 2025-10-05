import serial
import time
import threading
import sys

class ArduinoController:
    def __init__(self, port='COM3', baudrate=9600):
        """
        Initialize Arduino controller
        
        Args:
            port (str): Arduino COM port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate (int): Serial communication speed
        """
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.connected = False
        self.response_thread = None
        self.running = False
        
    def connect(self):
        """Connect to Arduino with verification"""
        try:
            print(f"Attempting to connect to Arduino on {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(3)  # Wait for Arduino to reset and initialize
            
            # Verify connection with handshake
            if self.verify_connection():
                self.connected = True
                self.running = True
                print(f"✓ Successfully connected to Arduino on {self.port}")
                return True
            else:
                print("✗ Arduino not responding properly")
                if self.arduino:
                    self.arduino.close()
                return False
                
        except serial.SerialException as e:
            print(f"✗ Failed to connect to Arduino: {e}")
            print("\nTroubleshooting steps:")
            print("1. Check if Arduino is connected to the correct port")
            print("2. Ensure port is not being used by another application")
            print("3. Close Arduino IDE Serial Monitor if open")
            print("4. Try a different COM port")
            print("5. Check if Arduino is properly powered")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
    
    def verify_connection(self):
        """Verify Arduino is responding properly"""
        try:
            # Clear any existing data
            self.arduino.flushInput()
            self.arduino.flushOutput()
            
            # Send status command and wait for response
            self.arduino.write(b"OK\n")
            time.sleep(0.5)
            
            # Read response with timeout
            response = ""
            start_time = time.time()
            while time.time() - start_time < 2:
                if self.arduino.in_waiting > 0:
                    response += self.arduino.read(self.arduino.in_waiting).decode('utf-8', errors='ignore')
                    if "OK_ACK" in response:
                        print(f"Arduino Response: {response.strip()}")
                        return True
                time.sleep(0.1)
            
            print(f"No valid response from Arduino. Received: {response}")
            return False
            
        except Exception as e:
            print(f"Connection verification failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        self.running = False
        
        if self.response_thread and self.response_thread.is_alive():
            self.response_thread.join(timeout=1)
        
        if self.arduino and self.connected:
            try:
                pass  # No reset command needed
            except:
                pass
            
            self.arduino.close()
            self.connected = False
            print("✓ Disconnected from Arduino")
    
    def send_signal(self, signal):
        """Send OK or NG signal to Arduino with acknowledgment"""
        if not self.connected:
            print("✗ Arduino not connected!")
            return False
        
        signal = signal.upper().strip()
        if signal not in ['OK', 'NG']:
            print("✗ Invalid signal! Use 'OK' or 'NG' only.")
            return False
        
        try:
            # Send command
            command = f"{signal}\n"
            self.arduino.write(command.encode())
            print(f"→ Sent '{signal}' signal to Arduino")
            
            # Wait for acknowledgment
            ack_received = False
            start_time = time.time()
            
            while time.time() - start_time < 1:  # 1 second timeout
                if self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    if response:
                        print(f"← Arduino: {response}")
                        if f"{signal}_ACK" in response:
                            ack_received = True
                            break
                time.sleep(0.01)
            
            if not ack_received:
                print(f"⚠ Warning: No acknowledgment received for {signal} command")
                return False
            
            return True
            
        except Exception as e:
            print(f"✗ Error sending signal: {e}")
            return False
    
    def get_status(self):
        """Get current pin status from Arduino"""
        if not self.connected:
            print("✗ Arduino not connected!")
            return False
        
        try:
            self.arduino.write(b"STATUS\n")
            time.sleep(0.2)
            
            response = ""
            start_time = time.time()
            while time.time() - start_time < 1:
                if self.arduino.in_waiting > 0:
                    response += self.arduino.read(self.arduino.in_waiting).decode('utf-8', errors='ignore')
                    if "STATUS:" in response:
                        print(f"← {response.strip()}")
                        return True
                time.sleep(0.01)
            
            print("No status response received")
            return False
            
        except Exception as e:
            print(f"✗ Error getting status: {e}")
            return False
    
    def reset_pins(self):
        """Reset both pins to LOW immediately"""
        if not self.connected:
            print("✗ Arduino not connected!")
            return False
        
        try:
            self.arduino.write(b"RESET\n")
            print("→ Reset command sent")
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"✗ Error resetting pins: {e}")
            return False
    
    def start_response_monitor(self):
        """Start background thread to monitor Arduino responses"""
        if self.response_thread and self.response_thread.is_alive():
            return
        
        def monitor_responses():
            while self.running and self.connected:
                try:
                    if self.arduino and self.arduino.in_waiting > 0:
                        response = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                        if response and not any(x in response for x in ["_ACK", "STATUS:"]):
                            print(f"← Arduino: {response}")
                except Exception as e:
                    if self.running:
                        print(f"Response monitor error: {e}")
                time.sleep(0.1)
        
        self.response_thread = threading.Thread(target=monitor_responses, daemon=True)
        self.response_thread.start()

def list_available_ports():
    """List available COM ports"""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if ports:
            print("\nAvailable COM ports:")
            for i, port in enumerate(ports, 1):
                print(f"  {i}. {port.device}: {port.description}")
            return ports
        else:
            print("No COM ports found!")
            return []
    except ImportError:
        print("Install pyserial for port detection: pip install pyserial")
        return []

def main():
    print("="*60)
    print("           ARDUINO PIN CONTROLLER")
    print("="*60)
    
    # List available ports
    available_ports = list_available_ports()
    
    # Configuration - You can change this default port
    ARDUINO_PORT = 'COM3'  # Change this to your Arduino's COM port
    
    if available_ports:
        print(f"\nUsing default port: {ARDUINO_PORT}")
        print("If this is incorrect, edit the ARDUINO_PORT variable in the code")
    
    controller = ArduinoController(ARDUINO_PORT)
    
    # Try to connect
    if not controller.connect():
        print("\n" + "="*60)
        print("CONNECTION FAILED - Please check your setup and try again")
        print("="*60)
        return
    
    # Start response monitoring
    controller.start_response_monitor()
    
    print("\n" + "="*60)
    print("           CONTROLLER READY")
    print("="*60)
    print("Commands:")
    print("  OK     - Turn Pin 4 HIGH for 2 seconds")
    print("  NG     - Turn Pin 5 HIGH for 2 seconds") 
    print("  quit   - Exit program")
    print("="*60)
    print("DEFAULT STATE: Both pins are LOW (FALSE)")
    print("Pin 4 & Pin 5 will go HIGH only when signals are received")
    print("="*60)
    
    try:
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit' or command == 'q':
                break
            elif command == 'ok':
                controller.send_signal('OK')
            elif command == 'ng':
                controller.send_signal('NG')
            elif command == '':
                continue
            else:
                print("✗ Invalid command! Use: OK, NG, or quit")
                
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    finally:
        print("\nShutting down...")
        controller.disconnect()
        print("Program ended")

if __name__ == "__main__":
    main()