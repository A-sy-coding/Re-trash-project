from sklearn.model_selection import train_test_split

def train_val_split(data): # 한 글래스의 전체경로 리스트가 들어오면 train, val 나누기
	train_data, valid_data = train_test_split(data, test_size=0.2, random_state = 0)

	return train_data, valid_data


