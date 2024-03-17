from model.SSD300 import SSD300
from model.FPN_SSD300_b import FPN_SSD300

model = SSD300()
total_params = sum(p.numel() for p in model.parameters())
print("Số lượng tham số của mô hình:", total_params)

