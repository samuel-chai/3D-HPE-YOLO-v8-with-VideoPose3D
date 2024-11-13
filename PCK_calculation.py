import pandas as pd
import numpy as np

# 設定閾值，以像素為單位（例如 10 像素）
PCK_THRESHOLD = 10

def calculate_pck(true_x, true_y, pred_x, pred_y, threshold=PCK_THRESHOLD):
    """
    計算 PCK（Percentage of Correct Keypoints）指標
    """
    distances = np.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
    correct_keypoints = distances < threshold
    pck_score = np.sum(correct_keypoints) / len(correct_keypoints)
    return pck_score

def main():
    # 讀取 Excel 文件
    file_path = "annotations.xlsx"  # 替換為您的文件路徑
    data = pd.read_excel(file_path)
    
    # 提取手動標註和模型標註的數據
    true_x, true_y = data['A'], data['B']
    model1_x, model1_y = data['C'], data['D']
    model2_x, model2_y = data['E'], data['F']
    
    # 計算兩個模型的 PCK
    pck_model1 = calculate_pck(true_x, true_y, model1_x, model1_y)
    pck_model2 = calculate_pck(true_x, true_y, model2_x, model2_y)
    
    # 輸出結果
    print(f"PCK of Model 1: {pck_model1 * 100:.2f}%")
    print(f"PCK of Model 2: {pck_model2 * 100:.2f}%")

if __name__ == "__main__":
    main()
