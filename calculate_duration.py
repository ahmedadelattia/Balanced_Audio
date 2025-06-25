#reads xlsx file and calculates the total duraiton of files in the file
import pandas as pd
import os
import soundfile as sf
import glob

def read_xlsx(file):
    df = pd.read_excel(file)
    return df
df = read_xlsx('balanced_subset.xlsx')

def obs_duration(obs_directory):
    max_duration = 0
    for cam in range(1,3):
        cam_duration = 0
        for file in glob.glob(obs_directory + f"/cam{cam}*.wav"):
            data, samplerate = sf.read(file)
            duration = len(data)/samplerate
            cam_duration += duration
        if cam_duration > max_duration:
            max_duration = cam_duration
    return max_duration

def calculate_duration(file):
    df = read_xlsx(file)
    duration = 0
    for index, row in df.iterrows():
        obs_dir = os.path.join("/media/ahmed/DATA 1/Research/Data/NCTE-Audio-Resampled_16KHz/", str(row['OBSID']))
        duration += obs_duration(obs_dir)
    return duration


if __name__ == "__main__":
    duration = calculate_duration('balanced_subset.xlsx')/3600
    
    with open('duration.txt', 'w') as f:
        print(duration, file=f)
    print("Duration calculated and saved to duration.txt")