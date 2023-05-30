from whoiswho.utils import load_json, save_json

#evaluation
def evaluate(assignment,ground_truth,type):
    if isinstance(assignment, str):
        assignment=load_json(assignment)
    if isinstance(ground_truth, str):
        ground_truth=load_json(ground_truth)

    submit_data=assignment

    result_list = []
    total_paper = 0

    # ground_truth 三级结构
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
    assignment = load_json('../training/result/result.valid.json')
    ground_truth = load_json('/home/hantianyi/whoiswho_dev/whoiswho/dataset/data/v3/RND/valid/cna_valid_ground_truth.json')
    evaluate(assignment,ground_truth)
