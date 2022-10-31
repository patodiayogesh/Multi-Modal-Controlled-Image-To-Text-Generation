from nltk.translate.bleu_score import sentence_bleu

def compute_bleu_scores(output, reference):

    bleu_4_sentence_scores = []
    for op, ref in zip(output, reference):

        bleu_4_sentence_scores.append(
            sentence_bleu(
                [x.split() for x in ref],
                op.split(),
                auto_reweigh=True,
            )
            * 100
        )
    return (
        sum(bleu_4_sentence_scores) / float(len(bleu_4_sentence_scores)),
        bleu_4_sentence_scores,
    )
