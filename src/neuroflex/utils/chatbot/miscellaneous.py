def keep_n_samples(
        dataset: Dataset,
        n: int,
        seed: int = seed
) -> Dataset:
    """
    Keeps only n samples of the Dataset given as input.

    Args:
        dataset (Dataset):
            The input dataset.
        n (int):
            The number of samples to keep.
        seed (int, optional):
            The seed value for random shuffling. Defaults to seed.

    Returns:
        Dataset: A new dataset containing only n samples.
    """

    # Shuffling the samples
    shuffled_indices = random.sample(range(len(dataset)), len(dataset))

    # Keeping only the first n samples
    shuffled_indices = shuffled_indices[:n]

    # Creating a new Dataset object containing the shuffled and truncated samples
    dataset_n_samples = dataset.select(shuffled_indices)

    return dataset_n_samples

def filter_samples(
    example: dict,
    filter_size: int
) -> bool:
    """
    Filters examples based on the presence of specific tokens and text length.

    Args:
        example (dict):
            The example to be filtered.
        filter_size (int):
            The maximum length allowed for the example text.

    Returns:
        bool:
            True if the example contains the token "### Assistant:" and its text length is less than filter_size, False otherwise.
    """
    return "### Assistant:" in example["text"] and len(example["text"]) < filter_size


# NOT USED
def preprocess_dataset_for_chatbot(
    dataset: DatasetDict,
    prompt_conversation_format: dict[str, dict[str, str]]
) -> DatasetDict:
    """
    Preprocesses the dataset to obtain the correct format.

    Args:
        dataset (DatasetDict):
            The dataset to be preprocessed.
        prompt_conversation_format (dict[str, dict[str, str]]):
            Dictionary containing formatting for user and chatbot inputs/outputs.

            Example format:
            {
                "user": {"start": tokenizer.bos_token + "[INST]" + tokenizer.eos_token, "end":"[\INST]"},
                "chatbot": {"start": tokenizer.bos_token, "end": tokenizer.eos_token}
            }

    Returns:
        DatasetDict:
            he preprocessed dataset in the correct format.
    """

    preprocessed_dataset = DatasetDict()
    for key in dataset.keys():
        single_set = []

        for elem in tqdm(dataset[key][:]["text"]):
            formatted_elem = prompt_conversation_format["user"]["start"] + elem[11:]
            formatted_elem = formatted_elem.replace("### Human:", prompt_conversation_format["chatbot"]["end"] + prompt_conversation_format["user"]["start"])
            formatted_elem = formatted_elem.replace("### Assistant:", prompt_conversation_format["user"]["end"] + prompt_conversation_format["chatbot"]["start"])
            formatted_elem = formatted_elem + prompt_conversation_format["chatbot"]["end"]

            single_set.append(formatted_elem)

        preprocessed_dataset[key] = Dataset.from_dict({"text": single_set})

    return preprocessed_dataset


def preprocess_function_for_chatbot(
    sentence: str,
    prompt_conversation_format: dict[str, dict[str, str]]
) -> str:
    """
    Preprocesses a conversation sentence for chatbot input.

    Args:
        sentence (str):
            The conversation sentence to preprocess.
        prompt_conversation_format (dict[str, dict[str, str]]):
            Dictionary containing formatting for user and chatbot inputs/outputs.

    Returns:
        str:
            The preprocessed conversation sentence formatted for chatbot input.
    """
    result = re.findall(r"(### Human: )(.*?)(?:### Assistant: (.*?)(?=### Human: |$)|$)", sentence, flags=re.DOTALL)

    new_sentence = ""
    for match in result:
        human_instruction = match[1]
        if match[2]:
            new_sentence += f"{prompt_conversation_format['user']['start']}{human_instruction.strip()}{prompt_conversation_format['user']['end']}{prompt_conversation_format['chatbot']['start']}{match[2].strip()}{prompt_conversation_format['chatbot']['end']}"
        else:
            new_sentence += f"{prompt_conversation_format['user']['start']}{human_instruction.strip()}{prompt_conversation_format['user']['end']}"

    return new_sentence.strip()


def find_last_subsequence_indexes(
        sequence: list,
        subsequence: list
) -> list[int]:
    """
    Finds the indexes of the last occurrence of a subsequence within a sequence.

    Args:
        sequence (list):
            The sequence to search within.
        subsequence (list):
            The subsequence to search for within the sequence.

    Returns:
        list[int]:
            A list containing the indexes of the last occurrence of the
            subsequence within the sequence.
    """

    subsequence_length = len(subsequence)

    # Iterate over the sequence in reverse
    for i in range(len(sequence) - subsequence_length, -1, -1):
        if sequence[i:i + subsequence_length] == subsequence:
            return i

    return len(sequence)


def cut_in_user(
        tokenizer: AutoTokenizer,
        tokenized_sentence: list[int],
        prompt_conversation_format: dict[str: dict[str: str]]
) -> bool:
    """
    Determines if the user is the last to speak in a conversation the given
    tokenized sentence.

    Args:
        tokenizer (AutoTokenizer):
            The tokenizer object used to tokenize prompts and sentences.
        tokenized_sentence (list[int]):
            The tokenized sentence to analyze.
        prompt_conversation_format (dict[str: dict[str: str]]):
            A dictionary containing the conversation format.

    Returns:
        bool:
            True if the user is the last to speak in a conversation the given
            tokenized sentence, False otherwise.
    """

    # Tokenizing user's start prompt
    tokenized_user_start = [elem for elem in tokenizer(prompt_conversation_format["user"]["start"]).input_ids[1:] if elem != 28705]
    # Tokenizing chatbot's start prompt
    tokenized_chatbot_start = [elem for elem in tokenizer(prompt_conversation_format["chatbot"]["start"]).input_ids[1:] if elem != 28705]
    # Finding last index of user's start prompt in tokenized sentence
    tokenized_user_start_index = find_last_subsequence_indexes(tokenized_sentence, tokenized_user_start)
    # Findinf last index of chatbot's start prompt in tokenized sentence
    tokenized_chatbot_start_index = find_last_subsequence_indexes(tokenized_sentence, tokenized_chatbot_start)

    """
    print(f"User tokenized start: {tokenized_user_start}")
    print(f"User tokenized start index: {tokenized_user_start_index}")
    print(len(tokenized_user_start))
    print(tokenized_sentence[tokenized_user_start_index:tokenized_user_start_index+len(tokenized_user_start)])
    print(tokenized_sentence[tokenized_user_start_index])
    print(f"Chatbot tokenized start: {tokenized_chatbot_start}")
    print(f"Chatbot tokenized start index: {tokenized_chatbot_start_index}")
    print(len(tokenized_chatbot_start))
    print(tokenized_sentence[tokenized_chatbot_start_index:tokenized_chatbot_start_index+len(tokenized_chatbot_start)])
    print(tokenized_sentence[tokenized_chatbot_start_index])
    """

    if find_last_subsequence_indexes(
        sequence = tokenized_user_start,
        subsequence = tokenized_chatbot_start
        ) < len(tokenized_user_start):

        if tokenized_user_start_index <= tokenized_chatbot_start_index < tokenized_user_start_index + len(tokenized_user_start):
            return True

    elif find_last_subsequence_indexes(
        sequence = tokenized_chatbot_start,
        subsequence = tokenized_user_start
        ) < len(tokenized_chatbot_start):

        if tokenized_chatbot_start_index <= tokenized_user_start_index < tokenized_chatbot_start_index + len(tokenized_chatbot_start):
            return False

    return tokenized_user_start_index > tokenized_chatbot_start_index


def tokenizer_function(
    examples: dict,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    preprocess: bool = True,
    prompt_conversation_format: dict = None
) -> dict:
    """
    Tokenizes the samples in the dataset.

    Args:
        examples (dict):
            Dictionary containing the examples to tokenize.
        tokenizer (AutoTokenizer):
            Tokenizer object.
        max_length (int):
            Maximum length of the tokenized sequences.
        preprocess (bool):
            Whether to preprocess the text before tokenization.
        prompt_conversation_format (dict):
            Dictionary containing formatting for user and chatbot inputs/outputs.

    Raises:
        Exception:
            If preprocessing is requested but the format is not provided.

    Returns:
        dict:
            Dictionary containing the tokenized inputs, labels and attention mask.
    """

    if preprocess:
        if prompt_conversation_format is None:
            raise Exception("If preprocessing is requested, the format must be provided")
        else:
            examples["text"] = [preprocess_function_for_chatbot(text, prompt_conversation_format) for text in examples["text"]]

    examples["text"] = [text[len(tokenizer.bos_token):] if text.startswith(tokenizer.bos_token) else text for text in examples["text"]]

    input_encodings = tokenizer(
        examples["text"],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )

    in_user = np.array([cut_in_user(tokenizer, input_encoding.tolist(), prompt_conversation_format) for input_encoding in input_encodings.input_ids])
    in_chatbot = ~in_user

    # Not changing the samples that are not cut
    in_user[np.array(input_encodings.input_ids[:, -1] == tokenizer(tokenizer.eos_token)["input_ids"][-1])] = False
    in_chatbot[np.array(input_encodings.input_ids[:, -1] == tokenizer(tokenizer.eos_token)["input_ids"][-1])] = False

    # Tokenizing user's end prompt
    tokenized_user_end = [elem for elem in tokenizer(prompt_conversation_format["user"]["end"]).input_ids[1:] if elem != 28705]
    # Tokenizing chatbot's end prompt
    tokenized_chatbot_end = [elem for elem in tokenizer(prompt_conversation_format["chatbot"]["end"]).input_ids[1:] if elem != 28705]

    input_encodings.input_ids[in_user, -len(tokenized_user_end):] = torch.LongTensor(tokenized_user_end)
    input_encodings.input_ids[in_chatbot, -len(tokenized_chatbot_end):] = torch.LongTensor(tokenized_chatbot_end)

    input_encodings.input_ids[:, -1] = tokenizer(tokenizer.eos_token)["input_ids"][-1]

    input_encodings["labels"] = input_encodings.input_ids.clone()
    input_encodings["labels"][input_encodings["attention_mask"] == 0] = -100

    examples["text"] = [tokenizer.bos_token + text for text in examples["text"]]

    return input_encodings

############################################################################################################################################################
#SPLIT TRAINING

def split_dataset_by_length(
    dataset: DatasetDict, interval_length_strings: int = 512,
    keys_to_split: list = ["train"], store_dataset: bool = True,
    preprocess_data: bool = True, format: dict = None, tokenize_data: bool = True,
    tokenizer = None, remove_text: bool = True
    ) -> DatasetDict:

  """
  """

  if tokenize_data and tokenizer is None:
    raise Exception("The tokenizer has to be set if the tokenizatino is requested (tokenize_data = True)")

  result = {}

  if preprocess_data:
    if format is None:
      raise Exception("format argument needs to be set in the case pre-processing is requested (preprocess_data = True)")
    # Preprocessing the dataset to obtain the correct format
    dataset = preprocess_dataset_for_chatbot(dataset, format)

  print()

  split_datasets = {}

  for key in dataset.keys():
    if key in keys_to_split:
      print(f"Splitting the strings in the dataset {key}")
      max_length = max(len(example) for example in dataset[key]["text"])
      k = int(np.ceil(max_length / interval_length_strings))

      for i in range(k):
        start_length = i * interval_length_strings
        end_length = (i + 1) * interval_length_strings
        print(f"Splitting the strings with length between {start_length} and {end_length}")

        if (start_length, end_length) not in split_datasets.keys():
          split_datasets[(start_length, end_length)] = {}

        if i < k-1:
          split_datasets[(start_length, end_length)][key] = dataset[key].filter(lambda example: start_length <= len(example["text"]) < end_length)
        else:
          split_datasets[(start_length, end_length)][key] = dataset[key].filter(lambda example: start_length <= len(example["text"]) <= end_length)

  dataset_dicts = {}
  for interval in split_datasets.keys():
    new_dataset_dict = DatasetDict()
    for key in split_datasets[interval].keys():
      new_dataset_dict[key] = split_datasets[interval][key]

    dataset_dicts[interval] = new_dataset_dict

    # Storing each split dataset in memory using pickle
    if store_dataset:
      with open(f"split_dataset_{interval}.pkl", "wb") as f:
        pickle.dump(new_dataset_dict, f)

  result['dataset_split'] = dataset_dicts

  print()

  if tokenize_data:
    dataset_dicts_tokenized = {}
    for interval in dataset_dicts.keys():
      max_length = interval[1] + 20

      print(f"Tokenizing the strings with length between {interval[0]} and {interval[1]}")

      # Tokenizing the dataset
      new_dataset_dict_tokenized = dataset_dicts[interval].map(lambda examples: tokenizer_function(examples, tokenizer, max_length), batched=True)

      if remove_text:
        for key in new_dataset_dict_tokenized.keys():
          new_dataset_dict_tokenized[key] = new_dataset_dict_tokenized[key].remove_columns(["text"])

      dataset_dicts_tokenized[interval] = new_dataset_dict_tokenized

      # Storing each split tokenized dataset in memory using pickle
      if store_dataset:
        with open(f"split_dataset_{interval}_tokenized.pkl", "wb") as f:
          pickle.dump(new_dataset_dict_tokenized, f)

    result["dataset_split_tokenized"] = dataset_dicts_tokenized

  return result

def stratified_split_by_length(dataset, interval_length_strings=512, test_size=0.1):
  # Create a dictionary to hold indices of samples for each interval
  interval_indices_dict = defaultdict(list)

  max_length = max(len(example) for example in dataset["train"]["text"])
  num_intervals = int(np.ceil(max_length / interval_length_strings))

  # Group indices by interval
  for i in range(num_intervals):
    interval_start = i * interval_length_strings
    interval_end = (i + 1) * interval_length_strings
    for idx, example in enumerate(dataset["train"]["text"]):
      if interval_start <= len(example) < interval_end:
        interval_indices_dict[i].append(idx)

  train_indices = []
  validation_indices = []

  # Randomly select one sample from each interval for both train and validation sets
  for interval_indices in interval_indices_dict.values():
    random.shuffle(interval_indices)
    train_indices.append(interval_indices.pop())
    validation_indices.append(interval_indices.pop())

  # Perform stratified split of remaining samples into train and validation based on the number of samples in each interval
  train_remaining_indices = []
  validation_remaining_indices = []
  for interval_indices in interval_indices_dict.values():
    train_interval_indices, val_interval_indices = train_test_split(interval_indices, test_size=test_size, stratify=[1] * len(interval_indices))
    train_remaining_indices.extend(train_interval_indices)
    validation_remaining_indices.extend(val_interval_indices)

  train_indices.extend(train_remaining_indices)
  validation_indices.extend(validation_remaining_indices)

  train_dataset = dataset["train"].select(train_indices)
  validation_dataset = dataset["train"].select(validation_indices)

  return train_dataset, validation_dataset

def load_datasets_from_pickle(
    file_prefix: str = 'split_dataset', interval_length_strings: int = 512, file_suffix: str = '_tokenized'
    ) -> dict:
  """
  Load tokenized dataset dictionaries from pickle files based on interval length.

  Parameters:
      file_prefix (str): The prefix used for the pickle files.
      interval_length_strings (int): The length of interval used for splitting the datasets.

  Returns:
      dict: A dictionary containing the loaded tokenized datasets.
  """

  tokenized_datasets = {}
  i = 0

  while True:
    file_name = f"{file_prefix}_({i * interval_length_strings}, {(i + 1) * interval_length_strings}){file_suffix}.pkl"
    if not os.path.exists(file_name):
      break

    with open(file_name, "rb") as f:
      print(f"Loading {file_name}")
      tokenized_datasets[(i * interval_length_strings, (i + 1) * interval_length_strings)] = pickle.load(f)

    i += 1

  return tokenized_datasets