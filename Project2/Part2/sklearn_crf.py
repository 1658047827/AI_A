import re


# 特征函数
def word2features_0(sent, language, i):
    word = sent[i]
    prev_word = "<start>" if i == 0 else sent[i - 1]  # START_TAG
    next_word = "<end>" if i == (len(sent) - 1) else sent[i + 1]  # STOP_TAG
    prev_word2 = "<start>" if i <= 1 else sent[i - 2]  # START_TAG
    next_word2 = "<end>" if i >= (len(sent) - 2) else sent[i + 2]  # STOP_TAG
    features = {
        "w": word,
        "w-1": prev_word,
        "w+1": next_word,
        "w-1:w": prev_word + word,
        "w:w+1": word + next_word,
        "w-1:w:w+1": prev_word + word + next_word,  # add
        "w-2:w": prev_word2 + word,  # add
        "w:w+2": word + next_word2,  # add
        "bias": 1,
        "word.isdigit": word.isdigit(),
    }
    return features


def word2features_1(sent, language, i):
    if language == "English":
        return en2features(sent, i)
    elif language == "Chinese":
        return word2features_0(sent, "", i)


def en2features(sent, i):
    word = sent[i]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "pos": sent[i],
        "word.length()": len(word),
        "word.isalnum()": word.isalnum(),
        "word.has_hyphen()": "-" in word,
        "word.has_digit()": any(char.isdigit() for char in word),
        # Add more features as needed
    }
    if i > 0:
        prev_word = sent[i - 1]
        features.update(
            {
                "prev_word.lower()": prev_word.lower(),
                "prev_word.isupper()": prev_word.isupper(),
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update(
            {
                "next_word.lower()": next_word.lower(),
                "next_word.isupper()": next_word.isupper(),
            }
        )
    else:
        features["EOS"] = True

    return features


def char_shape(char):
    if char.isnumeric():
        return "numeric"
    elif char.isalpha():
        if char.islower():
            return "lowercase"
        elif char.isupper():
            return "uppercase"
        else:
            return "mixedcase"
    else:
        return "other"


# 特征提取
def sent2features(sent, language, feature_func_num=0):
    if feature_func_num == 0:
        feature_func = word2features_0
    elif feature_func_num == 1:
        feature_func = word2features_1
    elif feature_func_num == 2:
        raise NotImplementedError
    return [feature_func(sent, language, i) for i in range(len(sent))]
