#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
import pickle
from mordred import Calculator, descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.feature_selection import VarianceThreshold
from itertools import compress
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

def standardize(smiles):
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    try: 
        mol = Chem.MolFromSmiles(smiles)

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol) 

        # if many fragments, get the "parent" (the actual mol we are interested in) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.

        te = rdMolStandardize.TautomerEnumerator() # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
     
        return  Chem.MolToSmiles(taut_uncharged_parent_clean_mol)
    
    except: 
        
        return "Cannot_do"

    
def calcdesc(data):
    # create descriptor calculator with all descriptors
    calc = Calculator(descriptors, ignore_3D=True)

    #print(len(calc.descriptors))
    Ser_Mol = data['smiles_r'].apply(Chem.MolFromSmiles)
    Mordred_table=  calc.pandas(Ser_Mol)
    Mordred_table = Mordred_table.astype('float')
    Mordred_table['smiles_r'] = data['smiles_r']
    
    Morgan_fingerprint= Ser_Mol.apply(GetMorganFingerprintAsBitVect, args=(2, 2048))
    Morganfingerprint_array  = np.stack(Morgan_fingerprint)

    Morgan_collection  = []
    for x in np.arange(Morganfingerprint_array.shape[1]): #np.arange plus rapide que range
        x = "Mfp"+str(x)
        Morgan_collection.append(x)

    Morganfingerprint_table  = pd.DataFrame(Morganfingerprint_array , columns=Morgan_collection )
    Morganfingerprint_table['smiles_r'] = data['smiles_r']
    
    data_mfp = pd.merge(data, Morganfingerprint_table)
    data_mfp_Mordred = pd.merge(data_mfp, Mordred_table)
    
    return(data_mfp_Mordred)
    
    
def calculate_similarity_test_vs_train(test, train):
    
    df_smiles_test = test['smiles_r']
    df_smiles_train = train['smiles_r']


    c_smiles_test = []
    for ds in df_smiles_test:
        try:
            cs = Chem.CanonSmiles(ds)
            c_smiles_test.append(cs)
        except:
            print("test")
            print('Invalid SMILES:', ds)



    c_smiles_train = []
    for ds in df_smiles_train:
        try:
            cs = Chem.CanonSmiles(ds)
            c_smiles_train.append(cs)
        except:
            print("train")
            print('Invalid SMILES:', ds)




    # make a list of mols
    ms_test = [Chem.MolFromSmiles(x) for x in c_smiles_test]

    # make a list of fingerprints (fp)
    fps_test = [AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048) for x in ms_test]

    # make a list of mols
    ms_train = [Chem.MolFromSmiles(x) for x in c_smiles_train]

    # make a list of fingerprints (fp)
    fps_train = [AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048) for x in ms_train]



    # the list for the dataframe
    qu, ta, sim = [], [], []

    # compare all fp pairwise without duplicates

    for a in range(len(fps_test)):

        s = DataStructs.BulkTanimotoSimilarity(fps_test[a], fps_train)    
        for m in range(len(s)):
            qu.append(c_smiles_test[a])
            ta.append(c_smiles_train[m])
            sim.append(s[m])

    # build the dataframe and sort it
    d = {'query':qu, 'target':ta, 'MFP_Tc':sim}
    df_final_ai= pd.DataFrame(data=d)
    return(df_final_ai)
    # save as csv
    

def determine_TS(test):

    train_data= pd.read_csv("./Train_data_log_transformed.csv")

    n_neighbours=5
    list_of_lists=[]
    df_master=pd.DataFrame()

    for endpoint in ["human_VDss_L_kg","human_CL_mL_min_kg", "human_fup", "human_mrt", "human_thalf"]:

        df = train_data
        df = df.dropna(subset=[endpoint]).reset_index(drop=True)
        df_final_ai = calculate_similarity_test_vs_train(test, df)
        df_final_ai = df_final_ai.sort_values('MFP_Tc', ascending=False)
        df_final_ai = df_final_ai.reset_index(drop=True)

        df_final_ai_2 = pd.DataFrame()
        for compound in df_final_ai["query"].unique():

            compounds_wise = pd.DataFrame()
            compounds_wise = df_final_ai[df_final_ai["query"]==compound].sort_values("MFP_Tc", ascending=False).iloc[:n_neighbours, :]
            df_final_ai_2 = pd.concat([df_final_ai_2, compounds_wise])

        df_final_ai_2 = df_final_ai_2.groupby('query').mean().sort_values("MFP_Tc").reset_index(drop=True)
        df_final_ai_2["endpoint"]=endpoint

        df_master = pd.concat([df_master, df_final_ai_2]).reset_index(drop=True)
        
    return(df_master.round(1))

def predict_individual_animal(data, endpoint, animal):#predict animal data
    
    #Read columns needed for rat data
    file = open(f"features_mfp_mordred_columns_{animal}_model.txt", "r")
    file_lines = file.read()
    features = file_lines.split("\n")
    features = features[:-1]

    loaded_rf = pickle.load(open(f"log_{endpoint}_model_FINAL.sav", 'rb'))

    X = data[features]
    y_pred =  loaded_rf.predict(X)   

    return(y_pred)

def predict_animal(data):
    
    endpoints = {"dog_VDss_L_kg","dog_CL_mL_min_kg","dog_fup"}

    for endpoint in endpoints:
        preds = predict_individual_animal(data, endpoint, "dog") 
        data[endpoint] = preds

    endpoints = {"monkey_VDss_L_kg","monkey_CL_mL_min_kg","monkey_fup"}

    for endpoint in endpoints:
        preds = predict_individual_animal(data, endpoint, "monkey") 
        data[endpoint] = preds
    
    endpoints = {"rat_VDss_L_kg","rat_CL_mL_min_kg","rat_fup"}

    for endpoint in endpoints:
        preds = predict_individual_animal(data, endpoint, "rat") 
        data[endpoint] = preds
    
    return(data)


def predict_VDss(data_mfp_Mordred, features_mfp_mordred_columns):#log human_VDss_L_kg model
    
    endpoint = "human_VDss_L_kg"
    features = features_mfp_mordred_columns

    loaded_rf = pickle.load(open("log_human_VDss_L_kg_withanimaldata_artificial_model_FINAL.sav", 'rb'))

    X = data_mfp_Mordred[features]
    y_pred =  loaded_rf.predict(X)   

    
    return("The VDss is", np.round(float(10**y_pred), 2), "L/kg")

def predict_CL(data_mfp_Mordred, features_mfp_mordred_columns):#log human_VDss_L_kg model
    
    endpoint = "human_VDss_L_kg"
    features = features_mfp_mordred_columns

    loaded_rf = pickle.load(open("log_human_CL_mL_min_kg_withanimaldata_artificial_model_FINAL.sav", 'rb'))

    X = data_mfp_Mordred[features]
    y_pred =  loaded_rf.predict(X)   

    
    #return("The CL is", np.round(float(10**y_pred), 2), "ml/min/kg")
    return(np.round(float(10**y_pred), 2))

#def predict_fup
#def predict_mrt
#def predict_thalf

def folderror_determiner(test):   
    baselines= pd.read_csv("fold_error_TS_allendpoints.csv")
    ts_data = determine_TS(test)
    fe_test_data = pd.merge(ts_data, baselines)
    return(fe_test_data)

def main():
    
    st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
    #st.image("Logo.png", width=100)
    st.title("PK Model")
    st.write(
    """
    [![Follow](https://img.shields.io/twitter/follow/srijitseal?style=social)](https://www.twitter.com/srijitseal)
    """
)
    
    smile=st.text_input("Enter SMILES")
    #smile="C#CCCCC(=O)c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1"
    
    smile = standardize(smile)
    test = {'smiles_r':  [smile]
            }
    test = pd.DataFrame(test)
    test_mfp_Mordred = calcdesc(test)

    folderror = folderror_determiner(test)

    #read from file features
    file = open("features_mfp_mordred_animal_artificial_human_modelcolumns.txt", "r")
    file_lines = file.read()
    features_mfp_mordred_animal_columns = file_lines.split("\n")
    features_mfp_mordred_animal_columns = features_mfp_mordred_animal_columns[:-1]

    #predict    
    pred = ''
    
    if st.button('Predict Human CL'):
        
        test_mfp_Mordred_animal = predict_animal(test_mfp_Mordred)
        pred = predict_CL(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
        
        fe = np.round(float(folderror[folderror["endpoint"]=="human_CL_mL_min_kg"]["folderror"]), 2)
        print("Expected range:", np.round(pred/fe,2), " to ", np.round(pred*fe, 2))
        
    st.success(pred)

if __name__ == '__main__': 
    main()   

