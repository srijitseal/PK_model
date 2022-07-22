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
    calc = Calculator(descriptors, ignore_3D=False)

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
    

def predict_individual_animal(data, endpoint, animal):#predict animal data
    
    #Read columns needed for rat data
    file = open(f"features_mfp_mordred_columns_{animal}_model.txt", "r")
    file_lines = file.read()
    features = file_lines.split("\n")
    features = features[:-1]

    loaded_rf = pickle.load(open(f"log_{endpoint}_model.sav", 'rb'))

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

    loaded_rf = pickle.load(open("log_human_VDss_L_kg_withanimaldata_model.sav", 'rb'))

    X = data_mfp_Mordred[features]
    y_pred =  loaded_rf.predict(X)   

    
    return("The VDss is", np.round(float(10**y_pred), 2), "L/kg")

def predict_CL(data_mfp_Mordred, features_mfp_mordred_columns):#log human_VDss_L_kg model
    
    endpoint = "human_VDss_L_kg"
    features = features_mfp_mordred_columns

    loaded_rf = pickle.load(open("log_human_CL_mL_min_kg_withanimaldata_model.sav", 'rb'))

    X = data_mfp_Mordred[features]
    y_pred =  loaded_rf.predict(X)   

    
    return("The CL is", np.round(float(10**y_pred), 2), "ml/min/kg")

def main():
    
    st.title("PK Model")

    smile=st.text_input("Enter SMILES")
    
    #smile="C#CCCCC(=O)c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1"
    
    smile = standardize(smile)
    data = {'smiles_r':  [smile]
        }
    data = pd.DataFrame(data)
    data_mfp_Mordred = calcdesc(data)
    

    #read from file features
    file = open("features_mfp_mordred_animal_human_modelcolumns.txt", "r")
    file_lines = file.read()
    features_mfp_mordred_animal_columns = file_lines.split("\n")
    features_mfp_mordred_animal_columns = features_mfp_mordred_animal_columns[:-1]
    
    #predict
    pred = ''
    
    if st.button('Predict Human VDss'):
        data_mfp_Mordred_animal = predict_animal(data_mfp_Mordred)
        pred = predict_VDss(data_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    
    if st.button('Predict Human CL'):
        data_mfp_Mordred_animal = predict_animal(data_mfp_Mordred)
        pred = predict_CL(data_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    st.success(pred)

if __name__ == '__main__': 
    main()        



