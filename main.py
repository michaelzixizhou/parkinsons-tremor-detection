from preprocessing import AccelerometerData
import numpy as np
import matplotlib.pyplot as plt

'''
if __name__ == "__main__":
    accelerometer_file_path = "data/accelerometer_data/801_1_accelerometer.pkl"
    accelerometer_data = AccelerometerData(accelerometer_file_path, frequency=100)
    accelerometer_data.plot_data()
    accelerometer_data.preprocess_data()
    accelerometer_data.plot_data()

'''

# Generate sample frequency-time domain data
def generate_sample_data(num_samples=1000, freq=100):
    t = np.linspace(0, num_samples / freq, num_samples)
    # Ensure the generated data is above 3.5 Hz
    data = np.array([
        np.abs(np.sin(2 * np.pi * 5 * t) + 3 * np.random.randn(num_samples)),  # X-axis
        np.abs(np.sin(2 * np.pi * 10 * t) + 1 * np.random.randn(num_samples)),  # Y-axis
        np.abs(np.sin(2 * np.pi * 15 * t) + 2 * np.random.randn(num_samples))   # Z-axis
    ])
    return data

# Test the functions
if __name__ == "__main__":
    frequency = 100  # 100 Hz sampling frequency

    # Create an instance of AccelerometerData
    acc_data = AccelerometerData(file_path=None, frequency=frequency)

    # Generate and set sample data
    acc_data.data = generate_sample_data()

    # Print original data
    print("Original Data:")
    acc_data.print_data()

    plt.plot(acc_data.data[0])
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Accelerometer Data")
    plt.show()

    # Preprocess data
    acc_data.preprocess_data()

    # Plot data
    plt.plot(acc_data.data)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Accelerometer Data")
    plt.show()
    