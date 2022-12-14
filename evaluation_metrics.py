from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text.rouge import ROUGEScore
# from evaluate import load

def compute_bleu_scores(output, reference):

    bleu_4_sentence_scores = []
    for op, ref in zip(output, reference):

        bleu_4_sentence_scores.append(
            round(sentence_bleu(
                [x.lower().split(' ') for x in ref],
                op.lower().split(' '),
                auto_reweigh=True,
            ) * 100, 2)
        )
    return (
        round(sum(bleu_4_sentence_scores) / len(bleu_4_sentence_scores), 2),
        bleu_4_sentence_scores,
    )

def compute_bert_score(output, reference):

    bert_scores = []
    #bertscorer = load("bertscore")
    for op, ref in zip(output, reference):

        ops = [op]*len(ref)
        bert_scores.append(max(
            #bertscorer.compute(predictions=ops, references=ref, model_type="bert-base-uncased")['precision']*100
        ))

    return (
        sum(bert_scores) / len(bert_scores),
        bert_scores,
    )

def compute_rouge_score(output, reference):

    rouge = ROUGEScore()
    scores = []
    for op, ref in zip(output, reference):
        temp_score = 0
        for r in ref:
            score = rouge(op, r)
            #print(score)
            rougeL_f_score = score['rougeL_fmeasure'].item()
            temp_score = max(temp_score, rougeL_f_score)
        scores.append(round(temp_score*100, 2))

    return (
        round(sum(scores)/len(scores), 2),
        scores
    )

def compute_meteor_score(output, reference):

    meteor_scores = []
    for op, ref in zip(output, reference):
        meteor_scores.append(
            round(meteor_score(
                [x.lower().split(' ') for x in ref],
                op.lower().split(' '),
            ) * 100, 2)
        )
    return (
        round(sum(meteor_scores) / len(meteor_scores), 2),
        meteor_scores,
    )

if __name__ == '__main__':

    preds = ['This is a test sentence', 'This is another test sentence']
    targets = [['This is test sentence', 'A test sentence'], ['This is test', 'A test sentence']]
    print(compute_bleu_scores(preds, targets))
    print(compute_rouge_score(preds, targets))
    print(compute_meteor_score(preds, targets))

