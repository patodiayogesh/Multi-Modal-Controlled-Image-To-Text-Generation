from nltk.translate.bleu_score import sentence_bleu
from evaluate import load

def compute_bleu_scores(output, reference):

    bleu_4_sentence_scores = []
    for op, ref in zip(output, reference):

        bleu_4_sentence_scores.append(
            sentence_bleu(
                [x.lower().split(' ') for x in ref],
                op.lower().split(' '),
                auto_reweigh=False,
            )
            * 100
        )
    return (
        sum(bleu_4_sentence_scores) / len(bleu_4_sentence_scores),
        bleu_4_sentence_scores,
    )

def compute_bert_score(output, reference):

    bert_scores = []
    bertscorer = load("bertscore")
    for op, ref in zip(output, reference):

        ops = [op]*len(ref)
        bert_scores.append(max(
            bertscorer.compute(predictions=ops, references=ref, model_type="bert-base-uncased")['precision']*100
        ))

    return (
        sum(bert_scores) / len(bert_scores),
        bert_scores,
    )
