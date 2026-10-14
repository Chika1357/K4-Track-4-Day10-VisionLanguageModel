"""Report observed empty/giant outputs after GRPO, without filtering predictions."""
import json
from pathlib import Path

from lab.reward import compute_reward
from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, parse_output


def analyze_bonus():
    records=read_jsonl('data/splits/test.jsonl')
    predictions={r['image_id']:r['raw'] for r in read_jsonl('work/e5.jsonl')}
    if set(predictions)!={r['image_id'] for r in records}:
        raise ValueError('Incomplete E5 predictions')
    empty=invalid=giant=positive_empty=empty_correct=0
    rewards=[]
    for record in records:
        raw=predictions[record['image_id']]
        parsed=parse_output(raw)
        truth=_gt_boxes(record)
        rewards.append(compute_reward(raw,truth))
        if parsed is None:invalid+=1
        elif not parsed:
            empty+=1
            if truth:positive_empty+=1
            else:empty_correct+=1
        elif any((p['bbox'][2]-p['bbox'][0])*(p['bbox'][3]-p['bbox'][1]) >= 800000 for p in parsed):
            giant+=1
    result={'n_images':len(records),'empty_outputs':empty,'empty_on_positive_images':positive_empty,
            'correct_empty_outputs':empty_correct,'invalid_outputs':invalid,'giant_box_images':giant,
            'giant_box_definition':'predicted area >= 80% of a norm1000 frame',
            'mean_test_reward':sum(rewards)/len(rewards)}
    Path('report/runs/e5_output_analysis.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    print(json.dumps(analyze_bonus(),indent=2))
