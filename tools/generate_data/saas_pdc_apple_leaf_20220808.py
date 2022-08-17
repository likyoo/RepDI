"""
20220808
自采数据,苹果叶片病害识别

"""

#对数据集生成固定格式的列表，格式为：图片的路径 <Tab> 图片类别的标签
import json
import os
import os.path as osp

def create_data_list(data_root_path, CLASS_LIST, train_dst_file_name, test_dst_file_name, readme_jsoin, division_ratio=5):
    with open(osp.join(data_root_path, test_dst_file_name), 'w') as f:
        pass
    with open(osp.join(data_root_path, train_dst_file_name), 'w') as f:
        pass
    # 所有类别的信息
    class_detail = []
    # 获取待识别的类别
    # class_dirs = os.listdir(data_root_path)
    class_dirs = CLASS_LIST
    # 类别标签
    class_label = 0
    # 获取总类别的名称
    father_paths = data_root_path.split('/')
    while True:
        if father_paths[len(father_paths) - 1] == '':
            del father_paths[len(father_paths) - 1]
        else:
            break
    father_path = father_paths[len(father_paths) - 1]

    all_class_images = 0
    other_file = 0
    # 读取每个类别
    for class_dir in class_dirs:
        if class_dir == test_dst_file_name or class_dir == train_dst_file_name or class_dir == readme_jsoin:
            other_file += 1
            continue
        print('正在读取类别：%s' % class_dir)
        # 每个类别的信息
        class_detail_list = {}
        test_sum = 0
        trainer_sum = 0
        # 统计每个类别有多少张图片
        class_sum = 0
        # 获取类别路径
        path = data_root_path + "/" + class_dir
        # 获取所有图片
        img_paths = os.listdir(path)
        for img_path in img_paths:
            # 每张图片的路径
            name_path = class_dir + '/' + img_path
            # 如果不存在这个文件夹,就创建
            if not os.path.exists(data_root_path):
                os.makedirs(data_root_path)
            # 划分训练集和测试集，各个类别中每隔 division_ratio 张选取一张作为测试集，并将数据集生成固定格式列表。
            if class_sum % division_ratio == 0:
                test_sum += 1
                with open(osp.join(data_root_path,test_dst_file_name), 'a') as f:
                    f.write(name_path + "\t%d" % class_label + "\n")
            else:
                trainer_sum += 1
                with open(osp.join(data_root_path, train_dst_file_name), 'a') as f:
                    f.write(name_path + "\t%d" % class_label + "\n")
            class_sum += 1
            all_class_images += 1
        # 说明的json文件的class_detail数据
        class_detail_list['class_name'] = class_dir
        class_detail_list['class_label'] = class_label
        class_detail_list['class_test_images'] = test_sum
        class_detail_list['class_trainer_images'] = trainer_sum
        class_detail.append(class_detail_list)
        class_label += 1
    # 获取类别数量
    all_class_sum = len(class_dirs) - other_file
    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = father_path
    readjson['all_class_sum'] = all_class_sum
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=-True, indent=4, separators=(',', ': '))
    with open(osp.join(data_root_path, readme_jsoin), 'w') as f:
        f.write(jsons)
    print('图像列表已生成')


#生成图像的列表
if __name__ == '__main__':
    # 把生产的数据列表都放在自己的总类别文件夹中
    data_root_path = "E:/dataset/果园病害图像_resize"
    train_dst_file_name = "train_saas_pdc_apple_leaf_20220808.list"
    test_dst_file_name = "test_saas_pdc_apple_leaf_20220808.list"
    readme_jsoin = "readme_saas_pdc_apple_leaf_20220808.json"
    CLASS_LIST = ['苹果健叶2021', '苹果白粉病', '苹果斑点落叶病', '苹果花叶病', '苹果缺素（黄化）', 
                  '苹果小叶病20220604', '苹果叶片褐斑病', '苹果叶片炭疽叶枯病', '苹果叶片锈病']
    division_ratio = 5 # 1/division_ratio的数据作为测试集
    create_data_list(data_root_path, CLASS_LIST, train_dst_file_name, test_dst_file_name, readme_jsoin, division_ratio)

