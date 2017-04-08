import pickle

def load_svc_scaler():
    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["X_scaler"]
    return svc, X_scaler

def save_svc_scaler(svc, X_scaler):
    dist_pickle = {"svc":svc,"X_scaler":X_scaler}
    pickle.dump(dist_pickle, open("svc_pickle.p","wb"))

    
def load_parameters():
    dist_pickle = pickle.load( open("params_pickle.p", "rb" ) )
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    return orient, pix_per_cell, cell_per_block,spatial_size,hist_bins
    
def save_parameters(h):
    params = h
    pickle.dump( params, open("params_pickle.p","wb"))


def pickle_data(x,y,name):
    dist_pickle = { }
    dist_pickle["features"] = x
    dist_pickle["labels"] = y
    pickle.dump(dist_pickle, open("{}.p".format(name), "wb")  )
    
def load_data(name):
    fname = "{}.p".format(name)

    try:
        with open(fname, mode='rb') as f:
            data = pickle.load(f)
        X,y = data["features"], data["labels"]
    except:
        return None
    
    return [X, y]
