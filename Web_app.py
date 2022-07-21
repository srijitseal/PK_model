{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "af10af7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "smile=\"C#CCN(C)C(C)Cc1ccccc1\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f9388c97",
   "metadata": {},
   "outputs": [],
   "source": [
    "from rdkit.Chem import inchi\n",
    "from rdkit import Chem\n",
    "from rdkit.Chem.MolStandardize import rdMolStandardize\n",
    "\n",
    "def standardize(smiles):\n",
    "    # follows the steps in\n",
    "    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb\n",
    "    # as described **excellently** (by Greg) in\n",
    "    # https://www.youtube.com/watch?v=eWTApNX8dJQ\n",
    "    try: \n",
    "        mol = Chem.MolFromSmiles(smiles)\n",
    "\n",
    "        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule\n",
    "        clean_mol = rdMolStandardize.Cleanup(mol) \n",
    "\n",
    "        # if many fragments, get the \"parent\" (the actual mol we are interested in) \n",
    "        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)\n",
    "\n",
    "        # try to neutralize molecule\n",
    "        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists\n",
    "        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)\n",
    "\n",
    "        # note that no attempt is made at reionization at this step\n",
    "        # nor at ionization at some pH (rdkit has no pKa caculator)\n",
    "        # the main aim to to represent all molecules from different sources\n",
    "        # in a (single) standard way, for use in ML, catalogue, etc.\n",
    "\n",
    "        te = rdMolStandardize.TautomerEnumerator() # idem\n",
    "        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)\n",
    "     \n",
    "        return  Chem.MolToSmiles(taut_uncharged_parent_clean_mol)\n",
    "    \n",
    "    except: \n",
    "        \n",
    "        return \"Cannot_do\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "abce8ba5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9c4b6fc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_selection import VarianceThreshold\n",
    "from itertools import compress\n",
    "\n",
    "def fs_variance(df, threshold:float=0.05):\n",
    "    \"\"\"\n",
    "    Return a list of selected variables based on the threshold.\n",
    "    \"\"\"\n",
    "\n",
    "    # The list of columns in the data frame\n",
    "    features = list(df.columns)\n",
    "    \n",
    "    # Initialize and fit the method\n",
    "    vt = VarianceThreshold(threshold = threshold)\n",
    "    _ = vt.fit(df)\n",
    "    \n",
    "    # Get which column names which pass the threshold\n",
    "    feat_select = list(compress(features, vt.get_support()))\n",
    "    \n",
    "    return feat_select\n",
    "\n",
    "def get_pairwise_correlation(population_df, method=\"pearson\"):\n",
    "    \"\"\"Given a population dataframe, calculate all pairwise correlations.\n",
    "    Parameters\n",
    "    ----------\n",
    "    population_df : pandas.core.frame.DataFrame\n",
    "        Includes metadata and observation features.\n",
    "    method : str, default \"pearson\"\n",
    "        Which correlation matrix to use to test cutoff.\n",
    "    Returns\n",
    "    -------\n",
    "    list of str\n",
    "        Features to exclude from the population_df.\n",
    "    \"\"\"\n",
    "\n",
    "\n",
    "    # Get a symmetrical correlation matrix\n",
    "    data_cor_df = population_df.corr(method=method)\n",
    "\n",
    "    # Create a copy of the dataframe to generate upper triangle of zeros\n",
    "    data_cor_natri_df = data_cor_df.copy()\n",
    "\n",
    "    # Replace upper triangle in correlation matrix with NaN\n",
    "    data_cor_natri_df = data_cor_natri_df.where(\n",
    "        np.tril(np.ones(data_cor_natri_df.shape), k=-1).astype(np.bool)\n",
    "    )\n",
    "\n",
    "    # Acquire pairwise correlations in a long format\n",
    "    # Note that we are using the NaN upper triangle DataFrame\n",
    "    pairwise_df = data_cor_natri_df.stack().reset_index()\n",
    "    pairwise_df.columns = [\"pair_a\", \"pair_b\", \"correlation\"]\n",
    "\n",
    "    return data_cor_df, pairwise_df\n",
    "\n",
    "def determine_high_cor_pair(correlation_row, sorted_correlation_pairs):\n",
    "    \"\"\"\n",
    "    Select highest correlated variable given a correlation row with columns:\n",
    "    [\"pair_a\", \"pair_b\", \"correlation\"]\n",
    "    For use in a pandas.apply()\n",
    "    \"\"\"\n",
    "\n",
    "    pair_a = correlation_row[\"pair_a\"]\n",
    "    pair_b = correlation_row[\"pair_b\"]\n",
    "\n",
    "    if sorted_correlation_pairs.get_loc(pair_a) > sorted_correlation_pairs.get_loc(pair_b):\n",
    "        return pair_a\n",
    "    else:\n",
    "        return pair_b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7715fadc",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "from rdkit import Chem\n",
    "from mordred import Calculator, descriptors\n",
    "import numpy as np\n",
    "from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect\n",
    "    \n",
    "def calcdesc(data):\n",
    "    # create descriptor calculator with all descriptors\n",
    "    calc = Calculator(descriptors, ignore_3D=False)\n",
    "\n",
    "    print(len(calc.descriptors))\n",
    "    Ser_Mol = data['smiles_r'].apply(Chem.MolFromSmiles)\n",
    "    Mordred_table=  calc.pandas(Ser_Mol)\n",
    "    Mordred_table = Mordred_table.astype('float')\n",
    "    Mordred_table['smiles_r'] = data['smiles_r']\n",
    "    \n",
    "    Morgan_fingerprint= Ser_Mol.apply(GetMorganFingerprintAsBitVect, args=(2, 2048))\n",
    "    Morganfingerprint_array  = np.stack(Morgan_fingerprint)\n",
    "\n",
    "    Morgan_collection  = []\n",
    "    for x in np.arange(Morganfingerprint_array.shape[1]): #np.arange plus rapide que range\n",
    "        x = \"Mfp\"+str(x)\n",
    "        Morgan_collection.append(x)\n",
    "\n",
    "    Morganfingerprint_table  = pd.DataFrame(Morganfingerprint_array , columns=Morgan_collection )\n",
    "    Morganfingerprint_table['smiles_r'] = data['smiles_r']\n",
    "    \n",
    "    data_mfp = pd.merge(data, Morganfingerprint_table)\n",
    "    data_mfp_Mordred = pd.merge(data_mfp, Mordred_table)\n",
    "    return(data_mfp_Mordred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "49799317",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import joblib\n",
    "\n",
    "def predict(data_mfp_Mordred, features_mfp_mordred_columns):#log human_VDss_L_kg model\n",
    "    \n",
    "    endpoint = \"human_VDss_L_kg\"\n",
    "    features = features_mfp_mordred_columns\n",
    "\n",
    "    loaded_rf = joblib.load(\"log_human_VDss_L_kg_model.joblib\")\n",
    "\n",
    "    X = data_mfp_Mordred[features]\n",
    "    y_pred =  loaded_rf.predict(X)   \n",
    "\n",
    "    \n",
    "    return(np.round(float(10**y_pred), 2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "6d7190cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "\n",
    "def main():\n",
    "    \n",
    "    st.title(\"PK Model\")\n",
    "\n",
    "    smile=st.text_input(\"Enter SMILES\")\n",
    "    \n",
    "    #smile=\"C#CCN(C)C(C)Cc1ccccc1\"\n",
    "    smile = standardize(smile)\n",
    "    data = {'smiles_r':  [smile]\n",
    "        }\n",
    "    data = pd.DataFrame(data)\n",
    "    \n",
    "    data_mfp_Mordred = calcdesc(data)\n",
    "\n",
    "    #read from file features\n",
    "    file = open(\"features_mfp_mordred_columns.txt\", \"r\")\n",
    "    file_lines = file.read()\n",
    "    features_mfp_mordred_columns = file_lines.split(\"\\n\")\n",
    "    features_mfp_mordred_columns = features_mfp_mordred_columns[:-1]\n",
    "    \n",
    "    #predict\n",
    "    \n",
    "    \n",
    "    if st.button('Predict Human VDss'):\n",
    "        VD= predict(data_mfp_Mordred, features_mfp_mordred_columns)\n",
    "    \n",
    "    st.success(VD)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "ec589b3b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1826\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 1/1 [00:00<00:00, 11.30it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/ss2686/miniconda3/envs/my-rdkit-env/lib/python3.9/site-packages/mordred/Autocorrelation.py:97: RuntimeWarning: Mean of empty slice.\n",
      "  return avec - avec.mean()\n",
      "/home/ss2686/miniconda3/envs/my-rdkit-env/lib/python3.9/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars\n",
      "  ret = ret.dtype.type(ret / rcount)\n",
      "/home/ss2686/miniconda3/envs/my-rdkit-env/lib/python3.9/site-packages/mordred/Constitutional.py:80: RuntimeWarning: invalid value encountered in double_scalars\n",
      "  return S / self.mol.GetNumAtoms()\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "ename": "UnboundLocalError",
     "evalue": "local variable 'VD' referenced before assignment",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mUnboundLocalError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_204086/3070175376.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'__main__'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m     \u001b[0mmain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/tmp/ipykernel_204086/1963276309.py\u001b[0m in \u001b[0;36mmain\u001b[0;34m()\u001b[0m\n\u001b[1;32m     27\u001b[0m         \u001b[0mVD\u001b[0m\u001b[0;34m=\u001b[0m \u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata_mfp_Mordred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfeatures_mfp_mordred_columns\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     28\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 29\u001b[0;31m     \u001b[0mst\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msuccess\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mVD\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     30\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mUnboundLocalError\u001b[0m: local variable 'VD' referenced before assignment"
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__': \n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ce02091",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
