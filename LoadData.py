import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import RobertaModel, RobertaTokenizer

mnli_train_data_size = -1
mnli_test_data_size = -1
train_data_size = 5000

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def prepare_glue_mnli(b_s):
    '''

    :param b_s:
    :return: 返回mnli task的train_loader
    '''

    dataset = load_dataset('glue', 'mnli')

    train_q1 = dataset['train']['premise'][:mnli_train_data_size]
    train_q2 = dataset['train']['hypothesis'][:mnli_train_data_size]

    train_e1 = tokenizer(train_q1, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    train_e2 = tokenizer(train_q2, padding='max_length', truncation=True, return_tensors='pt', max_length=128)

    train_labels = torch.LongTensor(dataset['train']['label'][:mnli_train_data_size])

    train_dataset = TensorDataset(train_e1['input_ids'], train_e1['attention_mask'], train_e2['input_ids'],
                                  train_e2['attention_mask'], train_labels)

    train_loader = DataLoader(train_dataset, batch_size=b_s, shuffle=True)

    test_q1 = dataset['validation_matched']['premise'][:mnli_test_data_size]
    test_q2 = dataset['validation_matched']['hypothesis'][:mnli_test_data_size]

    test_e1 = tokenizer(test_q1, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    test_e2 = tokenizer(test_q2, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    test_labels = torch.LongTensor(dataset['validation_matched']['label'][:mnli_test_data_size])

    test_dataset = TensorDataset(test_e1['input_ids'], test_e1['attention_mask'], test_e2['input_ids'],
                                 test_e2['attention_mask'], test_labels)

    test_loader = DataLoader(test_dataset, batch_size=b_s, shuffle=True)

    return train_loader, test_loader


def prepare_glue_qqp(b_s):
    '''

    :param b_s:
    :return: 返回qqp task的train_loader
    '''
    dataset = load_dataset('glue', 'qqp')

    train_q1 = dataset['train']['question1'][:train_data_size]
    train_q2 = dataset['train']['question2'][:train_data_size]
    train_e1 = tokenizer(train_q1, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    train_e2 = tokenizer(train_q2, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    train_labels = torch.FloatTensor(dataset['train']['label'][:train_data_size])
    train_dataset = TensorDataset(train_e1['input_ids'], train_e1['attention_mask'], train_e2['input_ids'],
                                  train_e2['attention_mask'], train_labels)
    train_loader = DataLoader(train_dataset, batch_size=b_s, shuffle=True)

    test_q1 = dataset['test']['question1'][:train_data_size]
    test_q2 = dataset['test']['question2'][:train_data_size]
    test_e1 = tokenizer(test_q1, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    test_e2 = tokenizer(test_q2, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    test_labels = torch.FloatTensor(dataset['test']['label'][:train_data_size])
    test_dataset = TensorDataset(test_e1['input_ids'], test_e1['attention_mask'], test_e2['input_ids'],
                                 test_e2['attention_mask'], test_labels)
    test_loader = DataLoader(test_dataset, batch_size=b_s, shuffle=True)
