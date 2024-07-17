import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sewar.full_ref import mse, rmse
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

cwd = r'C:\Users\evk77\OneDrive\Desktop\Prep Readings For MASc\Ni data\trail2'

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

def convert_tensor_string_to_list(tensor_string):
    tensor_string = tensor_string.replace('tensor([', '').replace('])', '')
    values = tensor_string.split(',')
    return [float(value.strip()) for value in values]

def process_batch_masks(batch_num):
    #data_mp = pd.read_csv(cwd + r'\mp_data.csv', index_col=0)
    #mask_mp = np.load(cwd + r'\mask.npy')
    data_mp = pd.read_csv(cwd+r'\mp_data_batch_{0}.csv'.format(batch_num), index_col=0)
    mask_mp = np.load(cwd+r'\mask_batch_{0}.npy'.format(batch_num))
    
    data_mp['xyxy'] = data_mp['xyxy'].apply(convert_tensor_string_to_list)
    resize_dim = (96, 96)
    masks_crop = []

    for idx, mask in enumerate(mask_mp):
        xyxy = data_mp['xyxy'].iloc[idx]
        angle = data_mp['Angle'].iloc[idx]
        x1 = float(xyxy[0])
        y1 = float(xyxy[1])
        x2 = float(xyxy[2])
        y2 = float(xyxy[3])
        centre = [(x1 + x2) / 2, (y1 + y2) / 2]

        mask_bw = mask.astype('uint8') * 255
        x1_crop = max(0, int(centre[0] - 275))
        x2_crop = min(mask.shape[1], int(centre[0] + 275))
        y1_crop = max(0, int(centre[1] - 275))
        y2_crop = min(mask.shape[0], int(centre[1] + 275))

        mask_crop = mask_bw[y1_crop:y2_crop, x1_crop:x2_crop]
        mask_crop_rs = cv2.resize(mask_crop, (resize_dim[1], resize_dim[0]))
        ret, mask_th = cv2.threshold(mask_crop_rs, 127, 255, cv2.THRESH_BINARY)
        mask_th[mask_th < 255] = 0
        masks_crop.append(mask_th)

    mask_mp_df = pd.DataFrame({'masks': masks_crop}).set_index(data_mp.index)
    data_mp_mask = pd.concat([data_mp, mask_mp_df], axis=1)

    return pd.DataFrame(data_mp_mask)

combined_df = process_batch_masks(1)

def drop_processed_data(df1, df2):
    combined = df1.merge(df2, on=['Power', 'Speed', 'rpm'], how='outer', indicator=True)
    df1_unique = combined[combined['_merge'] == 'left_only'].drop(columns=['_merge'])
    return df1_unique

# Combine each batch in to a single file - modify depending on how many batches there are 
df_b1 = process_batch_masks(1)
df_b2 = process_batch_masks(2)
df_b3 = process_batch_masks(3)
combined_df = pd.concat([df_b1, df_b2, df_b3])

# Drop the invalid data
df_to_drop = pd.read_csv(cwd+'\invalid_masks.csv')

def drop_processed_data(df1,df2): #Drop from df1, df2 is the rows to drop
    combined = df1.merge(df2, on=['Power', 'Speed', 'rpm'], how='outer', indicator=True)
    df1_unique = combined[combined['_merge'] == 'left_only'].drop(columns=['_merge'])
    return df1_unique

# Create the final dataset for training the model and save it for futher use
final_df = drop_processed_data(combined_df,df_to_drop)

#final_df = combined_df
final_df.to_csv(r'C:\Users\evk77\OneDrive\Desktop\Prep Readings For MASc\Ni data\trail2\processed_data_for_training_2.csv', index=False)

#corr = final_df[['Power', 'Speed', 'rpm', 'Width_mp', 'Depth', 'Dilutions']].corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)

X = final_df[['Power', 'rpm', 'Speed']]
y = final_df[['masks']]

y_np = np.stack(y['masks'].values)
resize_dim = (96, 96)
y_flat = y_np.reshape((y_np.shape[0], resize_dim[0] * resize_dim[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y_flat, test_size=0.2, random_state=42)

# train_indices = np.array([0, 1, 2, 5, 9, 14, 20, 26, 32, 38, 42, 45, 49, 53, 55, 60, 64, 68])  # Select specific indices for training data

# X = X.reset_index(drop=True)

# Select data based on indices
# X_train = X.loc[train_indices]
# y_train = y_flat[train_indices]

# Create test set with remaining data
# test_indices = np.setdiff1d(X.index, train_indices)
# X_test = X.loc[test_indices]
# y_test = y_flat[test_indices]

pca = PCA(n_components=0.95)
y_train_pca = pca.fit_transform(y_train)
y_test_pca = pca.transform(y_test)
y_recovered = pca.inverse_transform(y_train_pca)

n = 3
plt.figure(figsize=(n * 2, 4))
for i in range(1, n + 1):
    ax = plt.subplot(2, n, i)
    img_ori = y_train[i].reshape(resize_dim)
    plt.imshow(img_ori)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n)
    ret, imrecovered = cv2.threshold(y_recovered[i].reshape(resize_dim), 127, 255, cv2.THRESH_BINARY)
    plt.imshow(imrecovered)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.text(0, 80, 'MSE={:.2f}'.format(mse(imrecovered, img_ori)))
    plt.text(0, 90, 'RMSE={:.2f}'.format(rmse(imrecovered, img_ori)))
    plt.text(0, 100, 'P={0}, v={1}, rpm={2}'.format(X_train.iloc[i]['Power'], X_train.iloc[i]['Speed'], X_train.iloc[i]['rpm']), fontsize=8)
    
plt.suptitle('Meltpools before (top) and after (bottom) PCA', fontsize=16, x=0.5, y=0.95)

try:
    plt.savefig(r'C:\Users\evk77\OneDrive\Desktop\Prep Readings For MASc\Ni data\trail2\trained_models\PCAmp_2.svg', format='svg', dpi=300)
    plt.show()
except FileNotFoundError:
    os.mkdir(cwd + 'results')
    plt.savefig(cwd + 'trained_models/PCAmp.svg', format='svg', dpi=300)
    plt.show()

# =================== Model training with pre-trained model =============================== #
scaler_file = r'C:\Users\evk77\OneDrive\Desktop\Prep Readings For MASc\Ni data\trail2\sc.bin'
scaler = joblib.load(scaler_file)

#sc = StandardScaler()
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#pretrained_model_path = r'C:\Users\evk77\OneDrive\Desktop\Prep Readings For MASc\Ni data\SS data\trained_models\para2geom_transfer_learning_3264.h5'
pretrained_model_path = r'C:\Users\evk77\OneDrive\Desktop\Prep Readings For MASc\Ni data\trail2\para2geom.h5'
pretrained_model = load_model(pretrained_model_path)

# Freeze layers of the pre-trained model
for layer in pretrained_model.layers:
    layer.trainable = False
    print(f"Layer {layer.name} is trainable: {layer.trainable}")

# Add new layers for transfer learning
x = pretrained_model.layers[0].output  # Use only the output of the first layer
x = Dense(64, activation='relu', name='new_dense_1')(x)
#x = Dense(87, activation='relu', name='new_dense_2')(x)
x = Dense(y_train_pca.shape[1], activation='linear', name='new_dense_3')(x)

tf.random.set_seed(42)

new_model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=x)

# Compile the new model
new_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(patience=100, restore_best_weights=True)

# Train the new model
history = new_model.fit(X_train_std, y_train_pca, epochs=1000000, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Predict and evaluate the new model
y_pred_pca = new_model.predict(X_test_std)
y_pred = pca.inverse_transform(y_pred_pca)

# ======================== Results save ============================== #
# Save the model and the PCA transformer
new_model.save(cwd + r'\trained_models\para2geom_transfer_learning_2.h5')

# ======================== Results visualization ============================== #
n = min(10, len(y_test))  # Ensure n does not exceed the size of y_test
plt.figure(figsize=(n*2, 4))   

for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    img_ori = y_test[i].reshape(resize_dim)
    plt.imshow(img_ori)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    ret, imrecovered = cv2.threshold(y_pred[i].reshape(resize_dim), 127, 255, cv2.THRESH_BINARY)
    plt.imshow(imrecovered)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.text(0, 80, 'MSE={:.2f}'.format(mse(imrecovered, img_ori)))
    plt.text(0, 90, 'RMSE={:.2f}'.format(rmse(imrecovered, img_ori)))
    plt.text(0, 100, 'P={0}, v={1}, rpm={2}'.format(X_test.iloc[i]['Power'], X_test.iloc[i]['Speed'], X_test.iloc[i]['rpm']), fontsize=8)

plt.suptitle('Ground truth meltpools vs predicted meltpools', fontsize=16, x=0.7, y=0.95)
plt.savefig(cwd+r'\trained_models\MLP_mp_2.svg', format='svg', dpi=300)
plt.show()

# Parity Plot
true_areas = [np.sum(mask) for mask in y_test]
pred_areas = [np.sum(mask) for mask in y_pred]

plt.scatter(true_areas, pred_areas)
plt.xlabel('True Areas')
plt.ylabel('Predicted Areas')
plt.plot([min(true_areas), max(true_areas)], [min(true_areas), max(true_areas)], 'k--')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(cwd+r'\trained_models\MLP_Parity_2.svg', dpi=300)
plt.show()

r2 = r2_score(true_areas, pred_areas)
print(f'The R2 score for the model is: {r2:.3f}')