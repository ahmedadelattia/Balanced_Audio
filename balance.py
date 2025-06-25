import pandas as pd
import random
from time import time
import matplotlib.pyplot as plt
SIZE = 11
#replace with your direcotry 
df = pd.read_csv('transcript_metadata.csv')

# Display the first 5 rows
print(df.head())

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",

})
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

def calculate_imbalance(current_stats, target_distributions, weights = None):
    #TODO: imbalance as a dict to allow for weighted sum of imbalances
    imbalance_dict = {}
    
    for key in target_distributions:
        imbalance_dict[key] = {}
        for subkey in target_distributions[key]:
            imbalance_dict[key][subkey] = abs(current_stats[key][subkey] - target_distributions[key][subkey])
    if weights:
        
        for key in imbalance_dict:
            for subkey in imbalance_dict[key]:
                #if the key is not in the weights dict, it will be ignored (i.e. weight = 1)
                if key in weights:
                    if subkey in weights[key]:
                        imbalance_dict[key][subkey] *= weights[key][subkey]       
                
    imbalance = sum([sum(imbalance_dict[key].values()) for key in imbalance_dict])
        
    return imbalance

def optimize_subset(dataset, target_distributions, weights =None,  swap_size=1, size=100, min_iterations = 50000, patience = 0):
    current_subset = initialize_subset(dataset, size = size)
    current_loss = calculate_imbalance(compute_stats(current_subset), target_distributions, weights = weights)
    
    lowest_loss = current_loss
    start_time = time()
    num_iterations = 0
    curr_patience = patience
    losses = [current_loss]
    lowest_losses = [lowest_loss]
    for i in range(min_iterations):
        #TODO: Patience is buggy
        new_subset = current_subset.copy()
        to_remove = new_subset.sample(n=swap_size)
        to_add = dataset.loc[~dataset.index.isin(new_subset.index)].sample(n=swap_size)
        new_subset = pd.concat([new_subset.drop(to_remove.index), to_add])

        new_stats = compute_stats(new_subset)
        new_loss = calculate_imbalance(new_stats, target_distributions, weights = weights)
        losses.append(new_loss)
        #should I combine the two if statements? That would mean that the new set only gets saved if it is better by the percent_improvement. Leaving them separate can allow for a better set to be saved if it is only slightly worse
        if new_loss < current_loss:
            current_loss = new_loss
            current_subset = new_subset
            
        if num_iterations > min_iterations:
            if new_loss < lowest_loss:
                lowest_loss = new_loss
        lowest_losses.append(current_loss)

            #     curr_patience = patience
            # else:
            #     curr_patience -= 1
                
            # if curr_patience <= 0:
            #     print()
            #     print(f"Patience ran out, lowest loss = {lowest_loss}")
            #     break
        num_iterations += 1
        # if num_iterations % 100 == 0:
        curr_time_elapsed = time() - start_time
        time_per_iteration = curr_time_elapsed/num_iterations
        iteration_per_second = 1/time_per_iteration

        if curr_time_elapsed < 60:
            log_out = f"Time Elapsed: {curr_time_elapsed:.2f} seconds"
        elif curr_time_elapsed < 3600:
            log_out = f"Time Elapsed: {int(curr_time_elapsed/60)}:{int(curr_time_elapsed%60):02d} minutes"
        else:
            log_out = f"Time Elapsed: {int(curr_time_elapsed/3600)}:{int((curr_time_elapsed%3600)/60):02d}:{int(curr_time_elapsed%60):02d} hours"
        log_out += f", Iterations: {num_iterations}"
        if time_per_iteration > 1:
            log_out += f", Time per Iteration: {time_per_iteration:.2f} seconds"
        else:
            log_out += f", Iterations per Second: {iteration_per_second:.2f}"
        log_out += f", Current Loss: {new_loss:.2f}"
        if num_iterations > min_iterations:
            log_out += f", Lowest Loss: {lowest_loss:2f}, Patience: {curr_patience}"
        print(log_out, end='\r')
    print(f"Training done!")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label='Losses', color='skyblue')
    plt.plot(range(len(losses)), lowest_losses, label='Lowest Losses', color='orange')
    plt.xlabel(r'\textit{Iterations}', fontsize=16)
    # plt.ylabel('$\mathcal{L}_\text{Imbalance}$', fontsize=12)
    plt.ylabel(r'$\mathcal{L}_{Imbalance}$', fontsize=16)
    # plt.title('Losses over Iterations', fontsize=16, fontweight='bold')
    plt.legend(fontsize=16)
    # plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('losses_test.png')
    return current_subset


target_distributions = {
    'T_Gender': {'T_Male':1, 'T_Female': 0.5},
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

weights = {"S_Race": {'AFAM': 2, 'ASIAN': 2, "HISP": 2, 'WHITE': 2, "Others": 2},
           "T_Race": {'T_AFAM': 2, 'T_ASIAN': 2, 'T_HISP': 2, 'T_WHITE': 2,'T_Other': 2},
        }
balanced_subset = optimize_subset(df, target_distributions, size = SIZE)
final_stats = compute_stats(balanced_subset)
print(f"Final Stats: {final_stats}")
with open('stats.txt', 'a') as f:
    print(f"Final Stats: {final_stats}", file=f)
#Dumb balanced subset to xlsx
balanced_subset.to_excel('balanced_subset.xlsx')




