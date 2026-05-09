import sys
sys.path.append('.')

from lib import graph as graph_helper
import os
from multiprocessing.pool import Pool
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--graph_dir", type=str, help="input graph dir", default="data_self/graphs/vecroad_4/graphs_junc/"
)
parser.add_argument(
    "--save_dir", type=str, help="save wkt dir", default="data_self/graphs/vecroad_4/graphs_junc_wkt/"
)

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

def worker(f):
    print(f)
    name = f.split('.')[0]
    g = graph_helper.read_graph(os.path.join(args.graph_dir, f))
    g = g.clear_self()
    wkt = g.convert_rs_to_wkt()
    all_data = []
    for linestring in wkt:
        all_data.append(("AOI_0_{}_img0".format(name), linestring))
    df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
    df.to_csv(os.path.join(args.save_dir, name + '.csv'), index=False)

files = os.listdir(args.graph_dir)
pool = Pool()
pool.map(worker, files)
pool.close()
pool.join()

# 就是按照这个读取graph和写入CSV的方法，是没问题的，输出是对的（都很大一万多）
# 还是直接python eval/graph2wkt.py 默认参数是自己的路径
#
