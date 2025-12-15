import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import util as utils

def baca_data_csv(file):
    # Baca data_rumah
    return pd.read_csv(file)
    
def baca_data_xexcel(file):
    # Baca data_rumah
    return pd.read_excel(file)

def cek_data(data_rumah, konfig, api: bool = False):
    
    if not api:
        len_data_rumah = len(data_rumah)
    
        #cek tipe data
        assert data_rumah.select_dtypes("int").columns.to_list()[1:] == konfig["kolom_int"], "eror terjadi pada kolom int"
    
        #cek rentang data
        assert data_rumah[konfig["kolom_int"][0]].between(konfig["rentang_harga"][0], konfig["rentang_harga"][1]).sum() == len_data_rumah, "eror pada rentang harga"
        assert data_rumah[konfig["kolom_int"][1]].between(konfig["rentang_LB"][0], konfig["rentang_LB"][1]).sum() == len_data_rumah, "eror pada rentang Luas Bangunan"
        assert data_rumah[konfig["kolom_int"][2]].between(konfig["rentang_LT"][0], konfig["rentang_LT"][1]).sum() == len_data_rumah, "eror pada rentang Luas Tanah"
        assert data_rumah[konfig["kolom_int"][3]].between(konfig["rentang_KT"][0], konfig["rentang_KT"][1]).sum() == len_data_rumah, "eror pada rentang Jumlah Kamar Tidur"
        assert data_rumah[konfig["kolom_int"][4]].between(konfig["rentang_KM"][0], konfig["rentang_KM"][1]).sum() == len_data_rumah, "eror pada rentang Jumlah Kamar Mandi"
        assert data_rumah[konfig["kolom_int"][5]].between(konfig["rentang_GRS"][0], konfig["rentang_GRS"][1]).sum() == len_data_rumah, "eror pada rentang Garasi"
    else:
        pass
                                                      
        
if __name__ == "__main__":
    # 1. Muat file konfigurasi 
    konfig = utils.load_params(str(utils.dir_parent()) + utils.get_params())

    # 2. Baca data_rumah
    dir_data_raw = str(utils.dir_parent()) + utils.cek_path_os(konfig["dir_dataset"])
    data_rumah = baca_data_xexcel(dir_data_raw + konfig["file_xlsx"])
    
    # cek data defense
    cek_data(data_rumah, konfig)
    
    #konversi data ke pickel
    x = data_rumah[konfig["prediktor"]].copy()
    y = data_rumah[konfig["label"]].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 10)
    
    utils.pickle_dump(data_rumah, str(utils.dir_parent()) + utils.cek_path_os(konfig["dataset_cleaned_path"]))
    utils.pickle_dump(X_train, str(utils.dir_parent()) + utils.cek_path_os(konfig["train_set_path"][0]))
    utils.pickle_dump(y_train, str(utils.dir_parent()) + utils.cek_path_os(konfig["train_set_path"][1]))
    
    utils.pickle_dump(X_test, str(utils.dir_parent()) + utils.cek_path_os(konfig["test_set_path"][0]))
    utils.pickle_dump(y_test, str(utils.dir_parent()) + utils.cek_path_os(konfig["test_set_path"][1]))