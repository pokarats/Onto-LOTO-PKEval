import sys
import os
import platform
import re
import json
import argparse
from pathlib import Path
import logging
import time
from datetime import date
import traceback
from tqdm import tqdm
import numpy as np
from itertools import zip_longest
from smart_eval import matching_functions, scorer
sys.path.append(os.getcwd())
from src.utils.utils import CorpusIter, ProcessedIterExtended


PROJ_FOLDER = Path(__file__).resolve().parent.parent.parent

# Change the paths below as applicable to your file system
RESULT_DIR = PROJ_FOLDER / "res"
DATA_DIR = PROJ_FOLDER / "data"
MODEL_DIR = PROJ_FOLDER / "models" / "bleurt" / "BLEURT-20"


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', str(text))]


class EvalTextOCR:
    def __init__(self, ref_file_dir, can_file_dir, matching_type="bleurt", is_symmetric=False):
        self.matching_type = matching_type
        self.is_symmetric = is_symmetric
        self.ref_file_dir = ref_file_dir
        self.can_file_dir = can_file_dir
        self._references = {}
        self._candidates_a = {}
        self._candidates_b = {}
        self.all_scores = {}

    def _read_ref_file(self):

        for filepath in Path(self.ref_file_dir).iterdir():
            if filepath.is_dir():
                doc_name = str(filepath.name)
                self._references[doc_name] = []
                for file_subdir in sorted(filepath.glob("page_*"), key=natural_keys):
                    if file_subdir.is_dir():
                        for page_file in file_subdir.glob("*only_text.csv"):
                            if page_file.is_file():
                                logger.info(f"Reading ref file {doc_name} from page {file_subdir.name}")
                                page_text_iter = ProcessedIterExtended(page_file, True, ",", 9)
                                self._references[doc_name].append(page_text_iter)

        return None

    def _read_can_file(self, candidate_type):
        if candidate_type == "docling":
            logger.info(f"Reading {candidate_type} candidate files.")
            for filepath in Path(self.can_file_dir).iterdir():
                if filepath.is_dir():
                    doc_name = str(filepath.name)
                    self._candidates_a[doc_name] = []
                    for file_subdir in filepath.glob("docling_*"):
                        if file_subdir.is_dir():
                            for txt_subdir in file_subdir.glob("*txt_folder"):
                                for page_file in sorted(txt_subdir.glob("*_space_tag_rem.txt"), key=natural_keys):
                                    if page_file.is_file():
                                        logger.info(f"Reading candidate file {doc_name}: {page_file.name}")
                                        page_text_iter = CorpusIter(page_file)
                                        self._candidates_a[doc_name].append(page_text_iter)

        elif candidate_type == "pymu":
            logger.info(f"Reading {candidate_type} candidate files.")
            for folder_name in self._references.keys():
                self._candidates_b[folder_name] = []
                for filepath in (Path(RESULT_DIR) / folder_name).glob("data"):
                    logger.info(f"Reading from {filepath} candidate data dir")
                    if filepath.is_dir():
                        for page_file in sorted(filepath.glob("*_blocks_*.txt"), key=natural_keys):
                            if page_file.is_file():
                                logger.info(f"Reading candidate file {page_file.name}")
                                page_text_iter = CorpusIter(page_file)
                                self._candidates_b[folder_name].append(page_text_iter)

        else:
            raise NotImplementedError(f"{candidate_type} OCR type not implemented!")
        return None

    def _eval_page_content(self, reference_txt_iter, candidate_txt_iter):
        if self.matching_type == "bleurt":
            bleurt_chkpt = MODEL_DIR 
            logger.info(f"Running on platform {platform.system()}, "
                        f"loading BLERUT-20 model chkpt from {bleurt_chkpt}")
            mf = matching_functions.BleurtMatchingFunction(bleurt_chkpt.as_posix())
        else:
            mf = matching_functions.ChrfMatchingFunction()
        reference_txt = [txt_line for txt_line in reference_txt_iter]
        candidate_txt = [txt_line for txt_line in candidate_txt_iter]
        try:
            logger.info(f"ref_text last sent: {reference_txt[-1]}, len: {len(reference_txt)}")
        except IndexError:
            logger.error(f"ref_text len: {len(reference_txt)} means an empty list!\n"
                         f"Replacing with ['Empty Ref Sentence!!!']")
            reference_txt = ['Empty Ref Sentence!!!']
        try:
            logger.info(f"candidate_txt last sent: {candidate_txt[-1]}, len: {len(candidate_txt)}")
        except IndexError:
            logger.error(f"candidate_txt len: {len(candidate_txt)} means an empty list!\n"
                         f"Replacing with ['Empty Candidate Sentence!!!']")
            candidate_txt = ['Empty Candidate Sentence!!!']
        smart_scorer = scorer.SmartScorer(smart_types=['smart1', 'smartL'], matching_fn=mf, is_symmetric_matching=False)
        doc_scores_dict = smart_scorer.smart_score(reference=reference_txt, candidate=candidate_txt)

        return doc_scores_dict

    def _average_eval_scores(self, candidate_type):
        mean_score_dict = {'smart1': {},
                           'smartL': {}}

        for score_type in ['recall', 'precision', 'fmeasure']:
            mean_smart1_score = np.mean([dict_score['smart1'][score_type] for doc_name, list_dicts in self.all_scores[candidate_type].items() for dict_score in list_dicts])
            mean_score_dict['smart1'][score_type] = mean_smart1_score

            mean_smartL_score = np.mean([dict_score['smartL'][score_type] for doc_name, score_dict_list in self.all_scores[candidate_type].items() for dict_score in score_dict_list])
            mean_score_dict['smartL'][score_type] = mean_smartL_score

        logger.info(f"Average smart1 and smartL scores for {candidate_type}:\n{mean_score_dict}.")

        return mean_score_dict

    def evaluate(self, candidate_type):
        self.all_scores[candidate_type] = {}
        for doc_folder_name in self._references.keys():
            self.all_scores[candidate_type][doc_folder_name] = []
            logger.info(f"Evaluating {doc_folder_name} texts from {candidate_type} texts")
            if candidate_type == "docling":
                for pg_num, (ref_txt, can_txt) in enumerate(zip_longest(self._references[doc_folder_name],
                                                         self._candidates_a[doc_folder_name], fillvalue=["_xxx_"])):
                    page_eval_scores = self._eval_page_content(ref_txt, can_txt)
                    logger.info(f"{candidate_type} Doc page {pg_num} scores: {page_eval_scores}.")
                    self.all_scores[candidate_type][doc_folder_name].append(page_eval_scores)

            elif candidate_type == "pymu":
                for pg_num, (ref_txt, can_txt) in enumerate(zip_longest(self._references[doc_folder_name],
                                                         self._candidates_b[doc_folder_name], fillvalue=["_xxx_"])):
                    page_eval_scores = self._eval_page_content(ref_txt, can_txt)
                    logger.info(f"{candidate_type} Doc page {pg_num} scores: {page_eval_scores}.")
                    self.all_scores[candidate_type][doc_folder_name].append(page_eval_scores)
            else:
                raise NotImplementedError(f"Unsupported candidate type {candidate_type}!")

        return self._average_eval_scores(candidate_type)


def main(cl_args):
    start_time = time.time()

    ref_file_dr = RESULT_DIR / f"{cl_args.ref_filename}"
    can_file_dr = RESULT_DIR / f"{cl_args.can_filename}"
    nlg_eval_dir = RESULT_DIR / "OCR_NLG"
    if not nlg_eval_dir.exists():
        nlg_eval_dir.mkdir(parents=True, exist_ok=False)

    eval_ocr = EvalTextOCR(ref_file_dr, can_file_dr)
    eval_ocr._read_ref_file()
    eval_ocr._read_can_file(candidate_type="docling")
    eval_ocr._read_can_file(candidate_type="pymu")
    pymu_mean_scores = eval_ocr.evaluate(candidate_type="pymu")
    doc_ling_mean_scores = eval_ocr.evaluate(candidate_type="docling")


    for candidate_type, mean_scores in zip(["docling", "pymu"], [doc_ling_mean_scores, pymu_mean_scores]):
        output_filename = f"{can_file_dr.stem}_{cl_args.matching_type}_{candidate_type}_{cl_args.output_filename}"
        output_file_path = nlg_eval_dir / output_filename
        with open(output_file_path, "a") as out_file:
            json.dump(mean_scores, out_file, indent=1)
        logger.info(f"Saved scores to {output_file_path}")

    lapsed_time = (time.time() - start_time)
    time_minute = int(lapsed_time // 60)
    time_secs = int(lapsed_time % 60)
    logger.info(f"Script took {time_minute}:{time_secs} minutes!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_filename", action="store", type=str, default="smart_scores.json",
        help="Filename for the extracted contents WITH SUFFIX e.g. .md, .json, .txt"
    )
    parser.add_argument(
        "--ref_filename", action="store", type=str, default="Data_Files_Action_docling_txt_img_reformatted",
        help="Directory name containing the reference txt files"
    )
    parser.add_argument(
        "--can_filename", action="store", type=str, default="Data_Files_Action_docling_txt_img_reformatted",
        help="Directory name containing the candidate txt files"
    )
    parser.add_argument(
        "--matching_type", action="store", type=str, default="bleurt",
        help="Extraction type: bleurt, meteor, bleu, rouge etc. see google_research/smart_eval/matching_functions.py"
    )
    parser.add_argument(
        "--add_comment", action="store", type=str, default="smart_eval",
        help="Additional comment related to running the script"
    )

    parser.add_argument(
        "--quiet", action="store_true",
        help="Do not print logging to stdout."
    )

    args = parser.parse_args(sys.argv[1:])

    # Setup logging and start timer
    basename = Path(__file__).stem
    log_folder = PROJ_FOLDER / f".log/{date.today():%y_%m_%d}"
    log_file = log_folder / f"{time.strftime('%Hh%Mm%Ss')}_{basename}_{args.add_comment}.log"

    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=False)

    logging.basicConfig(format="%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s",
                        filename=log_file,
                        level=logging.INFO)

    # Manage the LOG and where to pipe it (log file only or log file + STDOUT)
    if not args.quiet:
        fmtr = logging.Formatter(fmt="%(funcName)s %(levelname)s: %(message)s")
        stderr_handler = logging.StreamHandler()
        stderr_handler.formatter = fmtr
        logging.getLogger().addHandler(stderr_handler)
        logging.info("Printing logging to the console")
    logger = logging.getLogger(__name__)
    logger.info(f"Running parameter \n{str(args.__dict__)}")

    try:
        main(args)
    except Exception as exp:
        if not args.quiet:
            print(f"Unhandled error: {repr(exp)}")
        logger.error(f"Unhandled error: {repr(exp)}")
        logger.error(traceback.format_exc())
        sys.exit(-1)
    finally:
        print(f"All Done, logged to {log_file}).")
