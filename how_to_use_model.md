추천 정보 갱신할 때 삭제할 것
1. ml-20m/pro_sg 폴더
2. ml-20m/reconstructed폴더
3. cf_list.csv 파일
4. model.pt 파일
---

1. data_cleansing 돌리면 딥러닝 모델 돌아감
2. cf돌리면 추천이되는거임.   문제점 -> 서버한테 유저아이디를 받아와야함.
				-> 파일 분할이 안되서 reconstructed 의 파일을 다시 만들 방안을 찾아야함(data_cleansing수정 필)
3. cf돌리면 cf_list.csv에 cf 유사도가 뜨고,
제일 높은 유저만 리턴하는 것도 가능. 추후 상의후 결정


4. 이 코드 서버로 옮기면 코드 내에 지정된 경로들 바꿔줘야함.
