import pytest

from lab.grpo import reward_function

ANSWER='[{"label":"pothole","bbox_2d":[100,100,200,200]}]'
GT=[{"label":"pothole","bbox":[100,100,200,200]}]


def test_string_and_chat_completions_receive_the_same_reward():
    assert reward_function([ANSWER,[{"role":"assistant","content":ANSWER}]], [GT,GT]) == [1.0,1.0]


def test_multimodal_text_completion_is_supported():
    completion=[{"role":"assistant","content":[{"type":"text","text":ANSWER}]}]
    assert reward_function([completion],[GT]) == [1.0]


def test_expanded_ground_truth_alignment_is_checked():
    with pytest.raises(ValueError,match="misaligned"):
        reward_function([ANSWER,ANSWER],[GT])
