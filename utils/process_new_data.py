# %%
from pathlib import Path
import os
import numpy as np
import pandas as pd

def Get_Data_Path(File):
    current_path = os.getcwd()
    Parent_Dir = str(Path(current_path).parents[0])
    data_path = os.path.join(Parent_Dir,'data')
    data_path = os.path.join(data_path,File)
    return data_path

def Get_DataFrame(data_path, type ):

    if type == 'SOC':
        new_data = pd.read_excel(data_path, sheet_name= None, header = None, names = ["soc", "person", "message"])   #pip install openpyxl before executing this command
        epistemic_data = pd.Series()
        social_data = pd.Series()
        message = pd.Series()

        for key in new_data:
            social_data = pd.concat([social_data, new_data[key]['soc']], ignore_index = True)
            message = pd.concat([message, new_data[key]['message']], ignore_index = True)

    else:
        new_data = pd.read_excel(data_path, sheet_name= None, header = None, names = ["person", "message", "epi"])   #pip install openpyxl before executing this command
        epistemic_data = pd.Series()
        social_data = pd.Series()
        message = pd.Series()

        for key in new_data:
            epistemic_data = pd.concat([epistemic_data, new_data[key]['epi']], ignore_index = True)
            message = pd.concat([message, new_data[key]['message']], ignore_index = True)


    new_data = pd.concat([social_data, epistemic_data, message], axis = 1)
    new_data.columns = ['soc', 'epi', 'message']
    return new_data

def Clean_Data(data):
    messages = []
    clean_soc = []
    clean_epi = []
    self_attr = ['R01', 'R02', 'R03', 'R04', 'R05', '1', '2', '3', '4', '5']
    parents_attr = ['R06', 'R07', 'R08', 'R09', 'R10', '6', '7', '8', '9', '10']
    teacher_attr = ['R11', 'R12', 'R13', 'R14', 'R15', '11', '12', '13', '14', '15']
    re_attr = ['R16', 'R17', '16', '17']
    cause_attr = ['R18', 'R19', 'R20', 'R21', '18', '19', '20', '21']

    for i in range(data.shape[0]):
        
        soc = data['soc'][i]
        epi = data['epi'][i]
        message = data['message'][i]
        if epi == epi:
            epi = str(epi)
            epi = epi.replace(",", " ")
            epi = epi.replace("\n", " ")
            epi = epi.split()
            for i in range(len(epi)):
                if epi[i] in self_attr:
                    epi[i] = '1'
                elif epi[i] in parents_attr:
                    epi[i] = '6'
                elif epi[i] in teacher_attr:
                    epi[i] = '11'
                elif epi[i] in re_attr:
                    epi[i] = '16'
                elif epi[i] in cause_attr:
                    epi[i] = '18'
                else:
                    epi[i] = '-1'
                
        clean_soc.append(soc)
        if epi == epi:
            cleaned_epi = list(set(epi))
            if '-1' in cleaned_epi and len(cleaned_epi) > 1:
                cleaned_epi.remove('-1')
            clean_epi.append(" ".join(cleaned_epi))
        else:
            clean_epi.append("-1")
        messages.append(message)                
    
    data = pd.DataFrame(data = [clean_soc, clean_epi, messages]).transpose()
    data.columns = ['soc', 'epi', 'message']
    return data


def Apply_Encoding(new_data):

    for i in range(len(new_data['soc'])):
        if new_data['soc'][i] == 'EX' or new_data['soc'][i] == 'Ex':
            new_data['soc'][i] = 35
        elif new_data['soc'][i] == 'EL':
            new_data['soc'][i] = 36
        elif new_data['soc'][i] == 'CON':
            new_data['soc'][i] = 37
        elif new_data['soc'][i] == 'Q':
            new_data['soc'][i] = 39
        elif new_data['soc'][i] == 'INT':
            new_data['soc'][i] = 41
        else:
            new_data['soc'][i] = -1

    return new_data

def Preprocess_New_Data(File, type):

    data_path = Get_Data_Path(File)
    new_data = Get_DataFrame(data_path, type)
    new_data = Clean_Data(new_data)
    new_data = Apply_Encoding(new_data)
    return new_data




# %%
