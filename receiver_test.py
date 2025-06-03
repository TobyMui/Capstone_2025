#!/usr/bin/env python3
"""
ArUco Shared Memory Reader
Reads position data from the ArUco detection system every 3 seconds
"""

import numpy as np
import time
import os
from multiprocessing import shared_memory


def read_position_data():
    """Read and parse position data from shared memory"""
    try:
        # Connect to existing shared memory
        position_mem = shared_memory.SharedMemory(name="aruco_position")

        # Read data from shared memory (7 float32 values = 28 bytes)
        data_bytes = bytes(position_mem.buf[:28])
        position_data = np.frombuffer(data_bytes, dtype=np.float32)

        # Parse the data: [found, center_x, center_y, frame_w, frame_h, error_x, error_y]
        found = bool(position_data[0])
        center_x = position_data[1]
        center_y = position_data[2]
        frame_w = position_data[3]
        frame_h = position_data[4]
        error_x = position_data[5]
        error_y = position_data[6]

        return {
            'found': found,
            'center_x': center_x,
            'center_y': center_y,
            'frame_width': frame_w,
            'frame_height': frame_h,
            'error_x': error_x,
            'error_y': error_y,
            'timestamp': time.time()
        }

    except FileNotFoundError:
        return {'error': 'Shared memory not found. Is the ArUco detection script running?'}
    except Exception as e:
        return {'error': f'Error reading shared memory: {str(e)}'}


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_position_data(data):
    """Format and print the position data"""
    clear_screen()
    print("ArUco Position Data Reader")
    print("Reading shared memory every 3 seconds... (Press Ctrl+C to exit)")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%H:%M:%S', time.localtime(data.get('timestamp', time.time())))}")

    if 'error' in data:
        print(f"ERROR: {data['error']}")
        return

    if data['found']:
        print("  MARKER DETECTED")
        print(f"  Position: ({data['center_x']:.1f}, {data['center_y']:.1f})")
        print(f"  Frame Size: {data['frame_width']:.0f} x {data['frame_height']:.0f}")
        print(f"  Error from Center: ({data['error_x']:+.1f}, {data['error_y']:+.1f})")

        # Calculate distance from center
        distance = np.sqrt(data['error_x'] ** 2 + data['error_y'] ** 2)
        print(f"  Distance from Center: {distance:.1f} pixels")

        # Provide positioning feedback
        if abs(data['error_x']) < 10 and abs(data['error_y']) < 10:
            print("  Status: 🟢 CENTERED")
        elif distance < 50:
            print("  Status: 🟡 CLOSE TO CENTER")
        else:
            print("  Status: 🔴 FAR FROM CENTER")

    else:
        print("   NO MARKER DETECTED")
        print(f"  Frame Size: {data['frame_width']:.0f} x {data['frame_height']:.0f}")


def main():
    """Main loop - read and print data every 3 seconds"""
    print("ArUco Position Data Reader")
    print("Reading shared memory every 3 seconds...")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            data = read_position_data()
            print_position_data(data)
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()