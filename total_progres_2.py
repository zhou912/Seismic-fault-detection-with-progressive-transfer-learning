#coding=utf-8
'''
该代码主要整合整个剖面处理流程：
1.训练剖面数据与需预测剖面生成
2.将含有标签的训练数据与不含标签的需预测数据一同使用迁移学习进行模型的训练用于预测剖面
3.在预测后使用人工图像处理增强模型的预测结果的连续性（这一步该如何做？使用迁移学习模型生成所有数据的预测结果，
然后将预测结果与原始数据共同输入至模型增加）
4.将预测出的结果当做标签，用于下一次训练的迭代
'''

#pre one section
import os
import data_make
import dann
import image_process
import scipy.io as io
from utils import *
import copy
import random
np.set_printoptions(threshold = np.inf)
from skimage.measure import compare_ssim
#数据初始化以及预处理


save_path = '/home/zrs/Desktop/zrs/dann_poumian_3/OUTPUT2'
save_path_2 = '/home/zrs/Desktop/zrs/dann_poumian_3/pre_data'
raw_data = data_make.data_normalize(io.loadmat('seismicdata.mat')['seismicdata'])
train_poumain_num_oral  = 10
n = 101
m = 401
train_x_oral,train_y_oral =  data_make.train_data_generate(train_poumain_num_oral)
train_x_oral = np.reshape(train_x_oral, (-1, 11, 11, 1))
raw_train_y = np.array(train_y_oral).astype(int)
train_y_oral = np.zeros((len(raw_train_y), 2))
train_y_oral[np.arange(len(raw_train_y)),raw_train_y] = 1
#记录整個工區訓練結果
total_final_pre = []
total_label = []

#備用訓練數據集合，當正在訓練的訓練集出現預測結果較差時，則返回使用預備訓練數據
train_x_reserve = [] 
train_y_reserve = []
train_id_reserve = []

#利用預測樣本生成的訓練數據集合，如果出現預測爲題則講該集合改變爲備用數據集合
train_x_being_used = []
train_y_being_used = []
train_id_being_used = []

#檢測預測結果異常的數據集合，如果最新的預測結果與該集合的差異過大，則認爲最新生成的預測結果存在一定的問題，
#更新預測數據集合，重新預測
pre_y_measure = [] 

#先使用前20個剖面進行預測完成各個集合的初始化，在初始化的過程中不涉及各個預測結果的判定
total_x_train = copy.deepcopy(train_x_oral)
total_y_train = copy.deepcopy(train_y_oral)
for i in range (20):
    print (i,'total_num')
    #预测剖面数据的生成
    deal_poumian_index = i + 1 + train_poumain_num_oral
    target_pre_x = data_make.pre_data_generate(deal_poumian_index,raw_data)
    target_pre_x = np.reshape(target_pre_x, (-1, 11, 11, 1))
    target_pre = dann.dann_total(total_x_train,total_y_train,target_pre_x,deal_poumian_index)
    output = np.reshape(target_pre,(n,m))
    total_final_pre.append(output)
    target_pre_image_progress = image_process.image_process(target_pre)
#    target_pre_image_progress = np.zeros((1,101,401))
    
    print (target_pre_image_progress.shape)
    total_label.append(target_pre_image_progress[0])

    add_y = []
    for l in range(n):
        for j in range(m):
            if target_pre_image_progress[0][l][j] > 0.6:
                add_y.append(1)
            else:
                add_y.append(0)
    add_y_mid = np.array(add_y).astype(int)
    add_y = np.zeros((len(add_y_mid), 2))
    add_y[np.arange(len(add_y_mid)), add_y_mid] = 1
    if i == 0 :
        train_x_reserve = target_pre_x
        train_y_reserve = add_y
        train_id_reserve.append(deal_poumian_index)
        
        train_x_being_used = target_pre_x
        train_y_being_used = add_y
        train_id_being_used.append(deal_poumian_index)
    elif i < 10 and i > 0 :
        train_x_reserve = np.concatenate((train_x_reserve,target_pre_x),axis=0)
        train_y_reserve = np.concatenate((train_y_reserve,add_y),axis=0)
        train_id_reserve.append(deal_poumian_index)
        
        train_x_being_used = np.concatenate((train_x_being_used,target_pre_x),axis=0)
        train_y_being_used = np.concatenate((train_y_being_used,add_y),axis=0)
        train_id_being_used.append(deal_poumian_index)
    else:
        delete_num = n * m
        train_x_being_used = train_x_being_used[delete_num:]
        train_y_being_used = train_y_being_used[delete_num:]
        train_x_being_used = np.concatenate((train_x_being_used,target_pre_x),axis=0)
        train_y_being_used = np.concatenate((train_y_being_used,add_y),axis=0)
        train_id_being_used = train_id_being_used[1:]
        train_id_being_used.append(deal_poumian_index)
    total_x_train = np.concatenate((train_x_oral,train_x_being_used),axis=0)
    total_y_train = np.concatenate((train_y_oral,train_y_being_used),axis=0)
    
    
#這個地方需要改成while 語句，循環的條件就是已經處理完的剖面數
deal_poumain = 0
similarity_index = True #用於衡量是否需要進行相似度判定：如果上一次相似度小於0.5，則使用備用訓練集
#進行訓練，此時就不會進行相似性度量，避免出現死循環


test_id_reserve = []
test_id_used = []
test_similarity = []
similarity_standard = 0.4
while deal_poumain < 340:
    print (deal_poumain + 20 , 'total_dealed_num')
    deal_poumian_index = deal_poumain + 21 + train_poumain_num_oral
    target_pre_x = data_make.pre_data_generate(deal_poumian_index,raw_data)
    target_pre_x = np.reshape(target_pre_x, (-1, 11, 11, 1))
    target_pre = dann.dann_total(total_x_train,total_y_train,target_pre_x,deal_poumian_index)
    #在此處需要進行當前預測結果與前幾個剖面的相似性度量，如果相似度較大則繼續正常運行
    #如果相似度較低則認定當前預測結果存在一定的問題需要重新訓練並預測
    similarity = 1
    if similarity_index == True:
        len_total = len(total_final_pre)
        #讀取了距離預測數據最近的五組數據
        pre_y_measure = total_final_pre[len_total - 5 :]
        pre_new = np.reshape(target_pre,(n,m))
        #求ssim與psnr
        SSIM_Total = 0
        PSNR_Total = 0
        for j in range (5):
            pre_y_measure_test = pre_y_measure[j]
            SSIM = compare_ssim(pre_new,pre_y_measure_test,data_range = 1)
            SSIM_Total = SSIM + SSIM_Total
            
#            diff = pre_new - pre_y_measure_test
#            mse = np.mean(np.square(diff))
#            PSNR = 10 * np.log10(1 * 1 / mse)

        SSIM_Ave = SSIM_Total/5
        similarity = SSIM_Ave
        
#        similarity = random.uniform(0.45,1)
        test_similarity.append(similarity)
        print (similarity)
    
    
    if (similarity > similarity_standard and similarity_index == True) or similarity_index == False:
        #這裏還要有一個指標來控制是否需要進行判定，如果上一次判定小於0.5，則返回使用備選集合裏面的
        #訓練數據，我們認爲備選集合裏面的訓練數據參考性很高，因此可以直接用於生成新的預測剖面
        output = np.reshape(target_pre,(n,m))
        total_final_pre.append(output)        
        target_pre_image_progress = image_process.image_process(target_pre)
#        target_pre_image_progress = np.zeros((1,101,401))
        total_label.append(target_pre_image_progress[0])
        
        
        add_y = []
        for l in range(n):
            for j in range(m):
                if target_pre_image_progress[0][l][j] > 0.6:
                    add_y.append(1)
                else:
                    add_y.append(0)
        add_y_mid = np.array(add_y).astype(int)
        add_y = np.zeros((len(add_y_mid), 2))
        add_y[np.arange(len(add_y_mid)), add_y_mid] = 1
        
        #先更新備用訓練數據集，然後更新正在使用的訓練集
        #先需要判定備用訓練數據集與正在使用的訓練集合是否存在交集
        intersection_index = list(set(train_id_reserve).intersection(set(train_id_being_used)))  
        delete_num = n * m
        if (len(intersection_index) == 0):
            train_x_reserve = train_x_reserve[delete_num:]
            train_y_reserve = train_y_reserve[delete_num:]
            train_id_reserve = train_id_reserve[1:]
            #觀察這裏添加的是不是40501
            x_reserve_add = train_x_being_used[:delete_num]
            y_reserve_add = train_y_being_used[:delete_num]
            id_reserve_add = train_id_being_used[0]
            
            train_x_reserve = np.concatenate((train_x_reserve,x_reserve_add),axis=0)
            train_y_reserve = np.concatenate((train_y_reserve,y_reserve_add),axis=0)
            train_id_reserve.append(id_reserve_add)
    
        train_x_being_used = train_x_being_used[delete_num:]
        train_y_being_used = train_y_being_used[delete_num:]
        train_x_being_used = np.concatenate((train_x_being_used,target_pre_x),axis=0)
        train_y_being_used = np.concatenate((train_y_being_used,add_y),axis=0)
        train_id_being_used = train_id_being_used[1:]
        train_id_being_used.append(deal_poumian_index)
        
        
        test_id_reserve.append(train_id_reserve)
        test_id_used.append(train_id_being_used)
        print (train_id_being_used,'train_id_being_used')
        print (train_id_reserve,'train_id_reserve')
        
        total_x_train = np.concatenate((train_x_oral,train_x_being_used),axis=0)
        total_y_train = np.concatenate((train_y_oral,train_y_being_used),axis=0)
        
        deal_poumain += 1
        similarity_index = True        
    elif similarity <= similarity_standard:
        similarity_index = False
        train_x_being_used =  train_x_reserve
        train_y_being_used =  train_y_reserve
        train_id_being_used = train_id_reserve
        print (train_id_being_used,'train_id_being_used')
        print (train_id_reserve,'train_id_reserve')
        test_id_reserve.append(train_id_reserve)
        test_id_used.append(train_id_being_used)
        continue
    

total_final_pre = np.array(total_final_pre)
total_label  = np.array(total_label)
np.save('test_final_pre_TFweightchange8',total_final_pre)
np.save('test_final_label_TFweightchange8',total_label)
print (total_final_pre.shape)

test_id_used = np.array(test_id_used)
test_id_reserve = np.array(test_id_reserve)
test_similarity = np.array(test_similarity)
print (test_similarity.shape,'test_similarity_shape15')
np.save('test_id_used_TFweightchange8',test_id_used)
np.save('test_id_reserve_TFweightchange8',test_id_reserve)
np.save('test_similarity_TFweightchange8',test_similarity)





