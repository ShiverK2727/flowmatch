import io
import os
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image




class CelebaHQDataset(Dataset):
    def __init__(self, parquet_path, transform=None, size=256):
        print(f"Loading dataset from: {parquet_path}")
        
        parquet_names = os.listdir(parquet_path)
        parquet_paths = [os.path.join(parquet_path, parquet_name) for parquet_name in parquet_names]
        # print(f"Found parquet files: {parquet_names}")

        if len(parquet_paths) < 1:
            return FileNotFoundError

        parquets = []

        for i in range(len(parquet_paths)):
            parquet_i = pd.read_parquet(parquet_paths[i])
            # print(f"Parquet file {parquet_paths[i]} columns: {parquet_i.columns.tolist()}")
            parquets.append(parquet_i)
            
        self.data = pd.concat(parquets, axis=0)
        # print(f"Combined dataset columns: {self.data.columns.tolist()}")
        self.size = size
        self.transform = transform

        # print(f"self.data: {self.data}")

    def __len__(self):
        
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_i = self.data.iloc[[idx]]
        
        # 获取图像数据
        image_data = data_i['image'].item()
        if isinstance(image_data, dict):
            if 'bytes' in image_data:
                image_data = image_data['bytes']
            else:
                # 如果没有bytes，尝试从path读取
                image_path = image_data.get('path')
                if image_path:
                    try:
                        image_i = Image.open(image_path).resize((self.size, self.size))
                        if self.transform is not None:
                            image_i = self.transform(image_i)
                        return image_i, data_i['label'].item()
                    except Exception as e:
                        print(f"Error reading image from path {image_path}: {e}")
                        raise e
                else:
                    raise ValueError("No bytes or path found in image data")
        
        try:
            # 确保数据是字节类型
            if isinstance(image_data, str):
                image_data = image_data.encode('utf-8')
            elif not isinstance(image_data, bytes):
                raise TypeError(f"Unexpected image data type: {type(image_data)}")
                
            image_i = Image.open(io.BytesIO(image_data)).resize((self.size, self.size))
        except Exception as e:
            print(f"Error processing image at index {idx}")
            print(f"Image data type: {type(image_data)}")
            raise e

        if self.transform is not None:
            image_i = self.transform(image_i)

        label_i = data_i['label'].item()

        return image_i, label_i



if __name__ == "__main__":
    dataset = CelebaHQDataset(parquet_path="/app/flowmatch/data/celeba-hq/data_split/val/")
    print(len(dataset))



