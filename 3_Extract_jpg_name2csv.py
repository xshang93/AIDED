import os
import csv
import re

def extract_data_from_filename(filename):
    # Using regular expressions to find the required numbers
    power_match = re.search(r'_P(\d+)_', filename)
    speed_match = re.search(r'_V(\d+)_', filename)
    rpm_match = re.search(r'_0_(\d+)', filename)

    power = power_match.group(1) if power_match else None
    speed = speed_match.group(1) if speed_match else None
    
    # Extract rpm and modify it according to the given conditions
    if rpm_match:
        rpm_value = int(rpm_match.group(1))
        if rpm_value in [1, 2]:
            rpm = 0.2
        elif rpm_value in [3, 4]:
            rpm = 0.4
        elif rpm_value in [5, 6]:
            rpm = 0.6
        elif rpm_value in [7, 8]:
            rpm = 0.8
        else:
            rpm = None  # or some default value if needed
    else:
        rpm = None


    return power, speed, rpm

def process_files_in_directory(directory):
    data = []
    for filename in os.listdir(directory):
        power, speed, rpm = extract_data_from_filename(filename)
        data.append([filename, power, speed, rpm])
    return data

def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Power', 'Speed', 'rpm'])  # Writing the header
        for row in data:
            writer.writerow(row)

directory_path = '/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/masked/invalid'  # Replace with your directory path
output_csv = '/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/invalid_masks.csv'  # Replace with your desired output file path

data = process_files_in_directory(directory_path)
write_to_csv(data, output_csv)

print(f"Data extracted and saved to {output_csv}")
