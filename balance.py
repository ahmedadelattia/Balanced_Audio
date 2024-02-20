import pandas as pd
import random
from tqdm import tqdm, trange
SIZE = 100
#replace with your direcotry 
df = pd.read_csv('transcript_metadata.csv')

# Display the first 5 rows
print(df.head())

def compute_stats(dataset):
    #TODO: Come up with a simplified logic to combine attibutes
    #ex add an arguments attr_to_combine
    #attr_to_combine = [['S_AFAM', 'S_Female'], ["T_Black", "T_Male"]]
    #Need also to check that combined attributes are not mutually exclusive
    teacher_male = dataset['MALE'].sum()/len(dataset)
    teacher_female = 1- teacher_male  #aka not male 
    T_Gender =  {'T_Male':teacher_male, 'T_Female':teacher_female}

    dataset['COUNT'] = dataset['S_MALE_count'] # Assume S_MALE_count is the number of individuals in each classroom 
    Stud_total = dataset['COUNT'].sum()
    eng_prof = (dataset['S_LEP']* dataset['COUNT']).sum()/ Stud_total
    S_Eng = {"1":eng_prof, '0': 1-eng_prof}
    student_male = (dataset['S_MALE']* dataset['COUNT']).sum()/ Stud_total
    S_Gender = {'Male':student_male, 'Female':1-student_male}
    

    AFAM = (dataset['S_AFAM']* dataset['COUNT']).sum()/ Stud_total
    ASIAN = (dataset['S_ASIAN']* dataset['COUNT']).sum()/ Stud_total
    HISP = (dataset['S_HISP']* dataset['COUNT']).sum()/ Stud_total
    WHITE = (dataset['S_WHITE']* dataset['COUNT']).sum()/ Stud_total
    Other = (dataset['S_RACE_OTHER'] * dataset['COUNT']).sum()/ Stud_total
    
    S_Race = {'AFAM':AFAM , 'ASIAN':ASIAN , "HISP":HISP , 'WHITE':WHITE, "Others":Other}


    T_AA = dataset['BLACK'].sum()/len(dataset)
    T_ASIAN = dataset['ASIAN'].sum()/len(dataset)
    T_HISP = dataset['HISP'].sum()/len(dataset)
    T_WHITE = dataset['WHITE'].sum()/len(dataset)
    T_AMINDIAN = dataset['AMINDIAN'].sum()/len(dataset)
    T_MULTIRACIAL = dataset['MULTIRACIAL'].sum()/len(dataset)
    T_Other = dataset['RACEOTHER'].sum()/len(dataset)
    T_RACE = {'T_AFAM':T_AA , 'T_ASIAN':T_ASIAN , "T_HISP":T_HISP , 'T_WHITE':T_WHITE, "T_AMINDIAN":T_AMINDIAN, "T_MULTIRACIAL":T_MULTIRACIAL, "T_Other":T_Other}



    stats =  {'S_Eng_Prof':S_Eng, 'S_Race':S_Race, 'S_Gender': S_Gender, 'T_Gender':T_Gender,'T_Race': T_RACE}
    return stats

stats = compute_stats(df)
print(f"Inital Stats: {stats}")
with open('stats.txt', 'w') as f:
    print(f"Inital Stats: {stats}", file=f)

def initialize_subset(dataset, size=100):
    return dataset.sample(n=size)

def calculate_imbalance(current_stats, target_distributions):
    imbalance = 0
    #TODO: imbalance as a dict to allow for weighted sum of imbalances
    for key in target_distributions:
        for subkey in target_distributions[key]:
            imbalance += abs(current_stats[key][subkey] - target_distributions[key][subkey])
        
    return imbalance

def optimize_subset(dataset, target_distributions, iterations=150000, swap_size=1, size=100):
    current_subset = initialize_subset(dataset, size = size)
    current_loss = calculate_imbalance(compute_stats(current_subset), target_distributions)
    pbar = trange(iterations)
    #TODO: Remove iterations and use a stopping criterion
    for _ in pbar:
        new_subset = current_subset.copy()
        to_remove = new_subset.sample(n=swap_size)
        to_add = dataset.loc[~dataset.index.isin(new_subset.index)].sample(n=swap_size)
        new_subset = pd.concat([new_subset.drop(to_remove.index), to_add])

        new_stats = compute_stats(new_subset)
        new_loss = calculate_imbalance(new_stats, target_distributions)
        
        if new_loss < current_loss:
            # print(f'loss = {calculate_imbalance(current_stats, target_distributions)}')
            current_loss = new_loss
            current_subset = new_subset
        pbar.set_description(f'loss = {current_loss:.2f}')
    pbar.close()
    print(f"Training done!")
    return current_subset


target_distributions = {
    'T_Gender': {'T_Male': 0.5, 'T_Female': 0.5},
    'T_Race': {'T_AFAM': 0.25, 'T_ASIAN': 0.25, 'T_HISP': 0.25, 'T_WHITE': 0.25,'T_Other': 0},
    'S_Eng_Prof': {'1': 0.5, '0': 0.5},
    'S_Race': {'AFAM': 0.25, 'ASIAN': 0.25, "HISP": 0.25, 'WHITE': 0.25, "Others": 0},
    'S_Gender': {'Male': 0.5, 'Female': 0.5}
}

#assert that the target distributions sum to 1
for key in target_distributions:
    assert sum(target_distributions[key].values()) == 1, f"Target distribution for {key} does not sum to 1"

print(f"Target Distributions: {target_distributions}")
with open('stats.txt', 'a') as f:
    print(f"Target Distributions: {target_distributions}", file=f)
    
balanced_subset = optimize_subset(df, target_distributions, size = SIZE)
final_stats = compute_stats(balanced_subset)
print(f"Final Stats: {final_stats}")
with open('stats.txt', 'a') as f:
    print(f"Final Stats: {final_stats}", file=f)
#Dumb balanced subset to xlsx
balanced_subset.to_excel('balanced_subset.xlsx')




