from whoiswho.utils import load_json, save_json

def evaluate(predict_result,ground_truth):
    if isinstance(predict_result, str):
        predict_result=load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth=load_json(ground_truth)

    submit_data=predict_result
    result_list = []
    total_paper = 0

    for name, authors in ground_truth.items():
        for a_id, papers in authors.items():
            predict_paper = set(submit_data.get(a_id, []))
            gt_papers = set(papers)

            inter_len = len(gt_papers & predict_paper)

            precision = round(inter_len / max(len(predict_paper), 1), 6)
            recall = round(inter_len / max(len(gt_papers), 1), 6)

            f1 = round(2 * precision * recall / max(precision + recall, 1))

            result_list.append((precision, recall, f1, len(gt_papers)))
            total_paper += len(gt_papers)

    # calculate weighted-f1
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    for instance in result_list:
        pre = instance[0]
        rec = instance[1]
        f1 = instance[2]
        weight = round(instance[3] / total_paper, 6)

        weighted_precision += pre * weight
        weighted_recall += rec * weight

    if (weighted_precision + weighted_recall) > 0:
        weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)

    print("f1: {} weighted precision: {} recall: {} ".format( weighted_f1,weighted_precision, weighted_recall))
    return weighted_f1

if __name__ == '__main__':
    predict_result = load_json('Input the path of result.valid.json')

    ground_truth = load_json('Input the path of cna_valid_ground_truth.json')
    evaluate(predict_result,ground_truth)
