import os
import torch
import numpy as np
import torch.utils.data
from utils import Add_Window_Horizon, StandardScaler

def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)
    # index_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    holiday_init = 1
    for index in range(data.shape[0]):
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if week_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None:
        k = 1
    else:
        for j in holiday_list :
            holiday_data[j-1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data

def load_st_dataset(dataset, args):
    if dataset == 'PEMS03':
        data_path = os.path.join('./data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    elif dataset == 'PEMS04':
        data_path = os.path.join('./data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    elif dataset == 'PEMS07':
        data_path = os.path.join('./data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    elif dataset == 'PEMS08':
        data_path = os.path.join('./data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    elif len(data.shape) > 2:
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    else:
        raise ValueError
    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
    return data

def normalize_dataset(data, normalizer, in_dim, column_wise=False):
    if normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            scaler = StandardScaler(mean, std)
            data[:, :, 0:in_dim] = scaler.transform(data[:, :, 0:in_dim])
        else:
            data_ori = data[:, :, 0:in_dim]
            data_day = data[:, :, in_dim:in_dim+1]
            data_week = data[:, :, in_dim+1:in_dim+2]

            mean_data = data_ori.mean()
            std_data = data_ori.std()
            mean_day = data_day.mean()
            std_day = data_day.std()
            mean_week = data_week.mean()
            std_week = data_week.std()

            scaler_data = StandardScaler(mean_data, std_data)
            data_ori = scaler_data.transform(data_ori)
            scaler_day = StandardScaler(mean_day, std_day)
            data_day = scaler_day.transform(data_day)
            scaler_week = StandardScaler(mean_week, std_week)
            data_week = scaler_week.transform(data_week)
            data = np.concatenate([data_ori, data_day, data_week], axis=-1)
            print(mean_data, std_data, mean_day, std_day, mean_week, std_week)
        print('Normalize the dataset by Standard Normalization')
    else:
        raise ValueError
    return data, scaler_data, scaler_day, scaler_week, None
    # return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(args, X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset, args)        # B, N, D
    #normalize st data
    # data, scaler = normalize_dataset(data, normalizer, args.column_wise)

    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)

    print('============', data_train.shape, data_val.shape, data_test.shape)
    _, scaler_data, scaler_day, scaler_week, scaler_holiday = normalize_dataset(data_train, normalizer, args.in_dim, args.column_wise)

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    x_tra_data = scaler_data.transform(x_tra[:, :, :, :args.in_dim])
    y_tra_data = scaler_data.transform(y_tra[:, :, :, :args.in_dim])
    x_tra_day = scaler_day.transform(x_tra[:, :, :, args.in_dim:args.in_dim+1])
    y_tra_day = scaler_day.transform(y_tra[:, :, :, args.in_dim:args.in_dim+1])
    x_tra_week = scaler_week.transform(x_tra[:, :, :, args.in_dim+1:args.in_dim+2])
    y_tra_week = scaler_week.transform(y_tra[:, :, :, args.in_dim+1:args.in_dim+2])
    x_tra = np.concatenate([x_tra_data, x_tra_day, x_tra_week], axis=-1)
    y_tra = np.concatenate([y_tra_data, y_tra_day, y_tra_week], axis=-1)

    x_val_data = scaler_data.transform(x_val[:, :, :, :args.in_dim])
    y_val_data = scaler_data.transform(y_val[:, :, :, :args.in_dim])
    x_val_day = scaler_day.transform(x_val[:, :, :, args.in_dim:args.in_dim+1])
    y_val_day = scaler_day.transform(y_val[:, :, :, args.in_dim:args.in_dim+1])
    x_val_week = scaler_week.transform(x_val[:, :, :, args.in_dim+1:args.in_dim+2])
    y_val_week = scaler_week.transform(y_val[:, :, :, args.in_dim+1:args.in_dim+2])
    x_val = np.concatenate([x_val_data, x_val_day, x_val_week], axis=-1)
    y_val = np.concatenate([y_val_data, y_val_day, y_val_week], axis=-1)

    x_test_data = scaler_data.transform(x_test[:, :, :, :args.in_dim])
    y_test_data = scaler_data.transform(y_test[:, :, :, :args.in_dim])
    x_test_day = scaler_day.transform(x_test[:, :, :, args.in_dim:args.in_dim+1])
    y_test_day = scaler_day.transform(y_test[:, :, :, args.in_dim:args.in_dim+1])
    x_test_week = scaler_week.transform(x_test[:, :, :, args.in_dim+1:args.in_dim+2])
    y_test_week = scaler_week.transform(y_test[:, :, :, args.in_dim+1:args.in_dim+2])
    x_test = np.concatenate([x_test_data, x_test_day, x_test_week], axis=-1)
    y_test = np.concatenate([y_test_data, y_test_day, y_test_week], axis=-1)

    ##############get dataloader######################
    train_dataloader = data_loader(args, x_tra, y_tra, args.batch_size, shuffle=True, drop_last=False)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(args, x_val, y_val, args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = data_loader(args, x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler_data, scaler_day, scaler_week, scaler_holiday