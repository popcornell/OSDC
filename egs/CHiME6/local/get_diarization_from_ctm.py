import yaml
import argparse
from parsing.get_diarization_from_ctm import get_diarization

parser = argparse.ArgumentParser("Parse ctm to get diarization in json file")
parser.add_argument("chime6_root", type=str)
parser.add_argument("split", type=str)
parser.add_argument("ctm_file", type=str)
parser.add_argument("out_folder", type=str)
parser.add_argument("conf_file", type=str)
parser.add_argument("--garbage_words", type=str, default="[inaudible]") #TODO


if __name__ == "__main__":

    args = parser.parse_args()
    with open(args.conf_file, "r") as f:
        confs = yaml.load(f)

    # test if compatible with lightning
    confs.update(args.__dict__)

    custom_json = confs["data"]["custom_json_folder"]
    get_diarization(confs["chime6_root"], confs["split"], confs["ctm_file"], confs["out_folder"], custom_json, garbage_words=["[inaudible]"])