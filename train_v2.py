"""
    Train Version 2
    Compute loss on branch nodes
    Compute accuracy on branch nodes

    Jason Yang
    March 5
"""
import traceback
from torch_geometric.data import DataLoader
from utils import *


def train(model, train_dataloader, valid_dataloader, optimizer, criterion, device):
    train_tot_epoch_loss = 0.0
    train_tot_epoch_accuracy = 0.0

    model.train()
    for i, data in enumerate(train_dataloader):
        features, edge_index, labels, batch = data.x, data.edge_index, data.y, data.batch

        features = features.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch = batch.to(device, non_blocking=True)

        flag1=False
        flag2=False

        optimizer.zero_grad()
        # [N,num_types]
        outputs = model(features, edge_index, batch)
        # print(outputs.shape)
        #================================================
        # label is br
        mask1 = (labels <= 2)
        maksed_labels_1 = labels[mask1]
        # make sure the labels contains br
        if maksed_labels_1.numel() != 0:
            maksed_output_1 = outputs[mask1]
            loss1= criterion(maksed_output_1, maksed_labels_1)
            loss1.backward(retain_graph=True)
            train_tot_epoch_loss += loss1.item()
            train_tot_epoch_accuracy += calc_accuracy(maksed_output_1, maksed_labels_1)
        #================================================
        # output is br
        output_idx=outputs.argmax(dim=-1)
        # output type is branch
        mask2 = (output_idx<=2)
        mask3 = (labels >2)
        output_filter=outputs[mask2 & mask3]
        if output_filter.numel() != 0:
            maksed_labels_2 = labels[mask2 & mask3]
            loss2= criterion(output_filter, maksed_labels_2)
            loss2.backward()
            train_tot_epoch_loss += loss2.item()
            train_tot_epoch_accuracy += calc_accuracy(output_filter, maksed_labels_2)
        #================================================
        if flag1 or flag2:
            optimizer.step()

        # TODO:
        #   should be checked
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():

        flag1=False
        flag2=False

        eval_tot_epoch_loss = 0.0
        eval_tot_epoch_accuracy = 0.0

        for i, data in enumerate(valid_dataloader):
            features, edge_index, labels, batch = data.x, data.edge_index, data.y, data.batch

            features = features.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch = batch.to(device,non_blocking=True)

            outputs = model(features, edge_index, batch)
            # print(outputs.shape)
            #================================================
            # label is br
            mask1 = (labels <= 2)
            maksed_labels_1 = labels[mask1]
            # make sure the labels contains br
            if maksed_labels_1.numel() != 0:
                maksed_output_1 = outputs[mask1]
                loss1= criterion(maksed_output_1, maksed_labels_1)

                eval_tot_epoch_loss += loss1.item()
                eval_tot_epoch_accuracy += calc_accuracy(maksed_output_1, maksed_labels_1)
            #================================================
            # output is br
            output_idx=outputs.argmax(dim=-1)
            # output type is branch
            mask2 = (output_idx<=2)
            mask3 = (labels >2)
            output_filter=outputs[mask2 & mask3]
            if output_filter.numel() != 0:
                maksed_labels_2 = labels[mask2 & mask3]
                loss2= criterion(output_filter, maksed_labels_2)
                eval_tot_epoch_loss += loss2.item()
                eval_tot_epoch_accuracy += calc_accuracy(output_filter, maksed_labels_2)


    return train_tot_epoch_loss / len(train_dataloader), \
           train_tot_epoch_accuracy / len(train_dataloader), \
           eval_tot_epoch_loss / len(valid_dataloader), \
           eval_tot_epoch_accuracy / len(valid_dataloader)


if __name__ == '__main__':

    config = PGOConfig()

    config.train_version = "v2"
    setup_seed(config.seed)
    device = config.device

    # TODO：
    #   add load checkpoint if needed
    ins_model = Model(
        in_features=config.in_features,
        hidden_features=config.hidden_features,
        out_features=config.out_features,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=0.5,
        conv_type=config.conv_layer,
        jkn_type=config.jkn_mode,
        activation_type=config.activation,
        glob_att=True
    ).to(device)

    if config.task == 0:
        criterion = config.loss_func_0
    else:
        criterion = config.loss_func_1

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(ins_model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(ins_model.parameters(), lr=config.lr)

    dataset = PGOGraphDataset(config.train_dataset_path)

    train_size = int(config.train_percent * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    dl_num_workers = config.dl_num_workers

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=2,
                                  shuffle=True,
                                  num_workers=dl_num_workers,
                                  pin_memory=True
                                  )
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=2,
                                  shuffle=True,
                                  num_workers=dl_num_workers,
                                  pin_memory=True)

    # log file directories
    dir_name = config.gen_config()
    this_log_dir = config.log_path + "/" + dir_name
    cmd1 = "mkdir -p " + this_log_dir
    do_shell_command_call(cmd1)
    # check points
    this_check_dir = config.log_path + "/" + dir_name + "/check_points"
    cmd2 = "mkdir -p " + this_check_dir
    do_shell_command_call(cmd2)

    file_metrics_path = this_log_dir + "/metrics.log"
    file_epoch_path = this_log_dir + "/epochs.log"
    file_config_path = this_log_dir + "/configs.log"
    file_res_path = this_log_dir + "/result.log"
    file_err_path = this_log_dir + "/err.log"

    checkpoint_path = this_check_dir

    # save config
    with open(file_config_path, "w") as fp:
        fp.write(dir_name + "\n")

    min_train_loss = float("inf")
    epoch_mtl = 0
    min_test_loss = float("inf")
    epoch_mtl2 = 0
    max_train_acc = float("-inf")
    epoch_Mtl = 0
    max_test_acc = float("-inf")
    epoch_Mtl2 = 0

    where = -1
    last_loss = float('inf')
    try:
        # for epoch in tqdm.tqdm(range(config.epochs)):
        for epoch in range(config.epochs):
            avg_epoch_train_loss, \
                avg_epoch_train_accuracy, \
                avg_epoch_valid_loss, \
                avg_epoch_valid_accuracy = train(
                ins_model,
                train_dataloader,
                valid_dataloader,
                optimizer,
                criterion,
                device)

            where = epoch
            last_loss = avg_epoch_train_loss
            # save metrics
            with open(file_metrics_path, "a") as fp:
                info1 = str(epoch) + "," + str(avg_epoch_train_loss) + "," + str(avg_epoch_train_accuracy) + "," + \
                        str(avg_epoch_valid_loss) + "," + str(avg_epoch_valid_accuracy) + "\n"
                fp.write(info1)

            # save epoch info
            info2 = "==============Epoch {}/{}==============\n" \
                    "train_loss: {:.4f}  train_accuracy:{:.4f}  test_loss: {:.4f}  test_accuracy:{:.4f}\n".format(
                epoch + 1,
                config.epochs,
                avg_epoch_train_loss,
                avg_epoch_train_accuracy,
                avg_epoch_valid_loss,
                avg_epoch_valid_accuracy)
            print(info2)
            with open(file_epoch_path, "a") as fp:
                fp.write(info2)

            if avg_epoch_train_loss < min_train_loss:
                min_train_loss = avg_epoch_train_loss
                epoch_mtl = epoch

            if avg_epoch_valid_loss < min_test_loss:
                min_test_loss = avg_epoch_valid_loss
                epoch_mtl2 = epoch

            if avg_epoch_train_accuracy > max_train_acc:
                max_train_acc = avg_epoch_train_accuracy
                epoch_Mtl = epoch

            if avg_epoch_valid_accuracy > max_test_acc:
                max_test_acc = avg_epoch_valid_accuracy
                epoch_Mtl2 = epoch

            # save result
            with open(file_res_path, "w") as fp:
                result1 = "min_train_loss:{} at epoch {}\n".format(min_train_loss, epoch_mtl)
                result2 = "min_test_loss:{} at epoch {}\n".format(min_test_loss, epoch_mtl2)
                result3 = "max_train_acc:{} at epoch {}\n".format(max_train_acc, epoch_Mtl)
                result4 = "max_test_acc:{} at epoch {}\n".format(max_test_acc, epoch_Mtl2)
                res = result1 + result2 + result3 + result4
                fp.write(res)
    except KeyboardInterrupt:
        print('manully stop training...')
    except Exception:
        # save err
        with open(file_err_path, "w") as fp:
            info = traceback.format_exc()
        print(info)
    finally:
        # save model checkpoint
        checkpoint_path = this_check_dir + "/" + 'model_checkpoint_{}.pth'.format(where + 1)
        torch.save({
            'epoch': where + 1,
            'model_state_dict': ins_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': last_loss
        }, checkpoint_path)
