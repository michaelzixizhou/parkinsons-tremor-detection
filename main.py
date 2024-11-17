from preprocessing import AccelerometerData

if __name__ == "__main__":
    accelerometer_file_path = "data/accelerometer_data/801_1_accelerometer.pkl"
    accelerometer_data = AccelerometerData(accelerometer_file_path, frequency=100)
    accelerometer_data.plot_data()
    accelerometer_data.preprocess_data()
    accelerometer_data.plot_data()
