import util as utils
import pandas as pd

def ambil_data():
    X_train = utils.pickle_load(str(utils.dir_parent()) + utils.cek_path_os(konfig["train_set_path"][0]))
    y_train = utils.pickle_load(str(utils.dir_parent()) + utils.cek_path_os(konfig["train_set_path"][1]))
    
    train_set = pd.concat([X_train, y_train], axis = 1)
    
    return train_set
    
def cek_outlier(dataset, pengkali):
    interkuartil = pengkali * (dataset["HARGA"].quantile(0.75) - dataset["HARGA"].quantile(0.25))
    batas_atas = dataset["HARGA"].quantile(0.75) + interkuartil
    data_outlier = dataset[dataset["HARGA"] > batas_atas]
    
    return data_outlier
    
def cek_data(dataset, kolom, rentang, pengkali):
    #fungsi ini awal2 akan memecah dataset dalam beberapa bagian sesuai rentang yang dimasukkan
    #kemudian akan dicek outliernya pada masing-masing blok tersebut
    data_outlier = pd.DataFrame()
    blok = int(dataset[kolom].max() / rentang)
    for i in range (1, blok):
        data_blok = dataset[dataset[kolom] < rentang * i]
        data_blok_cek_outlier = cek_outlier(data_blok, pengkali)
        data_outlier = pd.concat([data_outlier, data_blok_cek_outlier])
        #mengeluarkan data_blok dari dataset lama
        dataset = dataset[~dataset.index.isin(data_blok.index)]
    cek_outlier(dataset, pengkali)    
    return data_outlier
    
def hapus_outlier(dataset, outlier):
    try: 
        return dataset[~dataset.index.isin(outlier.index)]
    except:
        return dataset
    
if __name__ == "__main__":
    
    konfig = utils.load_params(str(utils.dir_parent()) + utils.get_params())
    
    train_set = ambil_data()
    
    outlier_LB = cek_data(train_set, "LB", konfig["blok_LB"][0], konfig["blok_LB"][1])
    outlier_LT = cek_data(train_set, "LT", konfig["blok_LT"][0], konfig["blok_LT"][1])
    
    train_set_clean = hapus_outlier(train_set, outlier_LB)
    train_set_clean = hapus_outlier(train_set_clean, outlier_LT)
    
    utils.pickle_dump(train_set_clean[konfig["prediktor"]], str(utils.dir_parent()) + utils.cek_path_os(konfig["train_clean_set_path"][0]))
    utils.pickle_dump(train_set_clean[konfig["label"]], str(utils.dir_parent()) + utils.cek_path_os(konfig["train_clean_set_path"][1]))

