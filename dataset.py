from sklearn.datasets import load_svmlight_file
import numpy as np
from joblib import Memory


class Dataset(object):
    def __init__(self, path=None, data=None, targets=None, cache=False, **kwargs):
        if path is not None:
            print(f"Reading dataset")
            if cache:
                mem = Memory("./mycache")
                @mem.cache
                def get_data(path):
                    data = load_svmlight_file(path)
                    return data[0], data[1]
                
                self.data, self.targets = get_data(path)
            else: 
                self.data, self.targets = load_svmlight_file(path)
            print(f"Finished reading.")
            try: 
                self.data = self.data.todense()
            except Exception:
                pass
            
            # Mechanism to prevent errors for some dataset (e.g., Suzy)
            inliers = np.where(np.linalg.norm(self.data, axis=1) < 1e15)[0]
            if len(inliers) < len(self.data):
                print(f"Removed {len(self.data) - len(inliers)} points because their norm was too large")
            self.data = self.data[inliers]
        
            
        elif data is not None and targets is not None:
            self.data = data
            self.targets = targets

        else: 
            print(f"Initiating empty Dataset object")
            return
         
        self.n, self.d = self.data.shape

    def split(self):
        med_target = np.median(self.targets)

        indices = list(np.where(self.targets < med_target)[0])
        d1 = Dataset(data=self.data[indices], targets=self.targets[indices])

        indices = list(np.where(self.targets >= med_target)[0])
        d2 = Dataset(data=self.data[indices], targets=self.targets[indices])

        return d1, d2
    
    def random_split(self):
        sep = int(self.n / 2)
        indices = np.array(range(self.n))
        np.random.shuffle(indices)
        indices = list(indices)

        d1 = Dataset(data=self.data[indices[:sep]], targets=self.targets[indices[:sep]])

        d2 = Dataset(data=self.data[indices[sep:]], targets=self.targets[indices[sep:]])

        return d1, d2
    
    def targets_split(self, thresh):
        unique_targets = list(set(self.targets))
        assert(len(unique_targets) == 2)

        indices = list(range(int(thresh / 2))) + list(np.where(self.targets[thresh:] == unique_targets[0])[0])
        d1 = Dataset(data=self.data[indices], targets=self.targets[indices])
        l1 = len(indices)

        indices = list(range(int(thresh / 2), thresh)) + list(np.where(self.targets[thresh:] == unique_targets[1])[0])
        d2 = Dataset(data=self.data[indices], targets=self.targets[indices])
        l2 = len(indices)

        print(f"Split dataset into two of sizes {l1} and {l2}")
        return d1, d2

    def truncate(self, n):
        return Dataset(data=self.data[:n], targets=self.targets[:n])
    
    def keep_indices(self, indices):
        return Dataset(data=self.data[indices], targets=self.targets[indices])
    

def merge_datasets(datasets):
    return Dataset(
        data=np.concatenate([d.data for d in datasets]),
        targets=np.concatenate([d.targets for d in datasets]) 
    ) 