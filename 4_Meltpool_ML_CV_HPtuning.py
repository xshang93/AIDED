import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sewar.full_ref import mse, rmse
import os

cwd = '/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/'
fig_save_dir = '/home/xiao/Dropbox/UofT/Project_docs/DED/Process_opt/figures/supplementary_figures/'

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

def convert_tensor_string_to_list(tensor_string):
    # Remove 'tensor(' and ')' from the string
    tensor_string = tensor_string.replace('tensor([', '').replace('])', '')
    
    # Split the string into individual values
    values = tensor_string.split(',')
    
    # Convert each value to a float and return as a list
    return [float(value.strip()) for value in values]

def process_batch_masks(batch_num):
    data_mp = pd.read_csv(cwd+'mp_data_batch_{0}.csv'.format(batch_num), index_col=0)
    mask_mp = np.load(cwd+'/mask_batch_{0}.npy'.format(batch_num))
    
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
        centre = [(x1+x2)/2, (y1+y2)/2]

        mask_bw = mask.astype('uint8')*255
        # Ensure cropping coordinates are within image dimensions.
        x1_crop = max(0, int(centre[0]-275))
        x2_crop = min(mask.shape[1], int(centre[0]+275))
        y1_crop = max(0, int(centre[1]-275))
        y2_crop = min(mask.shape[0], int(centre[1]+275))

        mask_crop = mask_bw[y1_crop:y2_crop, x1_crop:x2_crop]
        # mask_crop_rt = ndimage.rotate(mask_crop, np.rad2deg(angle))
        mask_crop_rs = cv2.resize(mask_crop, (resize_dim[1], resize_dim[0]))
        ret, mask_th = cv2.threshold(mask_crop_rs, 127, 255, cv2.THRESH_BINARY)
        mask_th[mask_th < 255] = 0
        masks_crop.append(mask_th)
        # plt.imshow(mask_th)
        # plt.savefig('/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/processed_masks/{0}.png'.format(idx))

    mask_mp_df = pd.DataFrame({'masks': masks_crop}).set_index(data_mp.index)
    data_mp_mask = pd.concat([data_mp, mask_mp_df], axis=1)

    return pd.DataFrame(data_mp_mask)

# Combine each batch in to a single file - modify depending on how many batches there are 
df_b1 = process_batch_masks(1)
df_b2 = process_batch_masks(2)
df_b3 = process_batch_masks(3)
df_b4 = process_batch_masks(4)
df_b5 = process_batch_masks(5)
combined_df = pd.concat([df_b1, df_b2, df_b3,df_b4,df_b5])

# Drop the invalid data
df_to_drop = pd.read_csv(cwd+'invalid_masks.csv')

def drop_processed_data(df1,df2): #Drop from df1, df2 is the rows to drop
    combined = df1.merge(df2, on=['Power', 'Speed', 'rpm'], how='outer', indicator=True)
    df1_unique = combined[combined['_merge'] == 'left_only'].drop(columns=['_merge'])
    return df1_unique

# Create the final dataset for training the model and save it for futher use
final_df = drop_processed_data(combined_df,df_to_drop)
final_df.to_csv(cwd+'processed_data_for_training.csv', index=False)

# final_df = final_df.dropna()
corr = final_df[['Power','Speed','rpm','Width_mp','Depth','Dilutions']].corr()
plt.figure(figsize=(10,10))
heatmap = sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns, annot=True)
plt.savefig(fig_save_dir+"heatmap_para2geom.svg", format='svg')

# # Looks like Power and Width are colinear. Keep both first and see. If not working well drop one.
X = final_df[[
    'Power',
    # 'Width',
    'rpm',
    'Speed',
    # 'Height',
    # 'Width',
    # 'Dilutions',
    # 'Depth'
    ]]

y = final_df[[
    'masks'
          ]]

# Determine the maximum size of the arrays in the DataFrame
y_np = np.stack(y['masks'].values)

resize_dim = (96,96)

# Dimension reduction for binary meltpools
y_flat = y_np.reshape((y_np.shape[0],resize_dim[0]*resize_dim[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y_flat, test_size=0.2, random_state=42)

pca_dim = PCA()
pca_dim.fit(y_train)
cumsum = np.cumsum(pca_dim.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 # d=87

pca = PCA(n_components=d)
y_train_pca = pca.fit_transform(y_train)
y_recovered = pca.inverse_transform(y_train_pca)

# Plots for meltpools before/after pca
n = 5
plt.figure(figsize=(n*2, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    img_ori = y_train[i].reshape(resize_dim)
    plt.imshow(img_ori)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    ret,imrecovered = cv2.threshold(y_recovered[i].reshape(resize_dim),127,255,cv2.THRESH_BINARY)
    plt.imshow(imrecovered)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.text(0,120,'P={0}, v={1}, rpm={2}'.format(X_train.iloc[i]['Power'],X_train.iloc[i]['Speed'],X_train.iloc[i]['rpm']),fontsize=8)
    
plt.suptitle('Meltpools before (top) and after (bottom) PCA',fontsize=16, x=0.5, y=0.95)

try:
    plt.savefig(fig_save_dir+'results/PCAmp.svg', format='svg',dpi=300)
    plt.show()
except FileNotFoundError:
    os.mkdir(fig_save_dir+'results')
    plt.savefig(fig_save_dir+'results/PCAmp.svg', format='svg',dpi=300)
    plt.show()    

# =================== Model training =============================== #
cv = False
model_type = 'nn'

# not performance cross validation (depreciated. This is not for pipeline)
if cv==False:
    
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # fit with only training data
    if model_type == 'nn':
        model = MLPRegressor(
            hidden_layer_sizes=(32,64),
            alpha=0.00001,
            learning_rate_init=0.0001,
            solver = 'adam',
            max_iter=1000000,
            n_iter_no_change=100,
            tol = 0.0001,
            verbose=True,
            random_state=42)
    elif model_type == 'lr':
        model = Ridge(random_state=42,max_iter=100000)
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
    
    model.fit(X_train_std,y_train_pca)
    y_pred_pca = model.predict(X_test_std)
    y_pred = pca.inverse_transform(y_pred_pca)
    
# Perform cross validation    
else:
    sc = StandardScaler()
    model = TransformedTargetRegressor(
        regressor = MLPRegressor(random_state=42,max_iter=1000000,n_iter_no_change=100,verbose=True),
    transformer = PCA(n_components=d),)
    pl = Pipeline(steps=[
        ('preprocessor',sc),
        ('estimater',model)
        ])
    # Create the parameter grid for hp tuning for ann
    param_grid = {
    'estimater__regressor__hidden_layer_sizes': [
        # (32),
                                                   (32,64),
        #                                          (32,64,88),
        #                                          (32,64,128,88),
                                                  # (32,64,128,256,128,88) # best config
                                                 ],
    'estimater__regressor__alpha': [
        # 0.00001,
        # 0.0001,
        # 0.001,
        0.1
        ],
    'estimater__regressor__learning_rate_init': [
        0.0001,
        0.001,
        0.01,
        0.1
        ]
    }

    grid_search = GridSearchCV(pl, param_grid = param_grid, 
                                cv = KFold(n_splits=5,shuffle=True, random_state=42), 
                               scoring = 'neg_root_mean_squared_error', 
                              n_jobs = 1, verbose = 3, 
                               refit = 'neg_root_mean_squared_error',
                              )
    # fit with full dataset
    grid_search.fit(X_train,y_train)
    model = grid_search.best_estimator_
    
    y_pred = model.predict(X_test)
    
    # Convert the cv_results_ to a DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    # Save the DataFrame to a CSV file
    results_df.to_csv('./trained_models/grid_search_results_32_alpha.csv', index=False)
    
    print(grid_search.best_params_)

# ======================== Results visualization ============================== #

# Plot results to compare ground true and predictions
n = 10
plt.figure(figsize=(n*2, 4))   

for i in range(6, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)

    img_ori = y_test[i].reshape(resize_dim)
    plt.imshow(img_ori)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n)
    ret,imrecovered = cv2.threshold(y_pred[i].reshape(resize_dim),127,255,cv2.THRESH_BINARY)
    plt.imshow(imrecovered)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plt.text(0,80,'MSE={:.2f}'.format(mse(imrecovered,img_ori)))
    # plt.text(0,90,'RMSE={:.2f}'.format(rmse(imrecovered,img_ori)))
    plt.text(0,120,'P={0}, v={1}, rpm={2}'.format(X_test.iloc[i]['Power'],X_test.iloc[i]['Speed'],X_test.iloc[i]['rpm']),fontsize=8)

plt.suptitle('Ground truth meltpools vs predicted meltpools',fontsize=16, x=0.7, y=0.95)     
plt.savefig(fig_save_dir+'results/MLP_mp.svg', format='svg',dpi=300)
plt.show()

# Parity Plot
# Comparing total contour areas
true_areas = [np.sum(mask) for mask in y_test]
pred_areas = [np.sum(mask) for mask in y_pred]

fig1,ax1 = plt.subplots(figsize=(4,4))

ax1.set_position([0.1, 0.1, 0.8, 0.8])

ax1.scatter(true_areas, pred_areas)
ax1.set_xlabel('True Areas')
ax1.set_ylabel('Predicted Areas')
ax1.plot([min(true_areas), max(true_areas)], [min(true_areas), max(true_areas)], 'k--')
# plt.scatter(true_areas, pred_areas)
# plt.xlabel('True Areas')
# plt.ylabel('Predicted Areas')
# plt.plot([min(true_areas), max(true_areas)], [min(true_areas), max(true_areas)], 'k--')
# plt.tight_layout()
r2 = r2_score(true_areas, pred_areas)
plt.text(0,110,'r2={:.4f}'.format(r2))
plt.savefig(fig_save_dir+'results/MLP_Parity.svg', dpi=300)
plt.show()

print(f'The R2 score for the model is: {r2:.3f}')

# # ======================== Results save ============================== #

# # save trained model for use
# joblib.dump(model, './trained_models/para2geom.pkl')
# joblib.dump(sc, './trained_models/std_scaler.bin', compress=True)

# # Save testing data for future use
# X_test.to_csv('./trained_models/para2geom_X_test.csv')
# np.save('./trained_models/para2geom_y_test',y_test)