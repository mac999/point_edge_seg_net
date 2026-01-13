def diagnose_test_issues(train_data, test_data, model):
    print("=== 테스트 문제 진단 ===")
    
    # 1. 데이터 형태 확인
    print(f"Train data shape: {train_data.x.shape}")
    print(f"Test data shape: {test_data.x.shape}")
    
    # 2. 특징 값 범위 확인  
    print(f"Train features range: {train_data.x.min():.3f} ~ {train_data.x.max():.3f}")
    print(f"Test features range: {test_data.x.min():.3f} ~ {test_data.x.max():.3f}")
    
    # 3. 위치 좌표 범위 확인
    print(f"Train pos range: {train_data.pos.min():.3f} ~ {train_data.pos.max():.3f}")
    print(f"Test pos range: {test_data.pos.min():.3f} ~ {test_data.pos.max():.3f}")
    
    # 4. 레이블 분포 확인
    train_labels = torch.unique(train_data.y, return_counts=True)
    test_labels = torch.unique(test_data.y, return_counts=True)
    print(f"Train label distribution: {train_labels}")
    print(f"Test label distribution: {test_labels}")
    
    # 5. 모델 출력 확인
    model.eval()
    with torch.no_grad():
        train_output = model(train_data[:1000])  # 샘플 체크
        test_output = model(test_data[:1000])
        
    print(f"Train output range: {train_output.min():.3f} ~ {train_output.max():.3f}")  
    print(f"Test output range: {test_output.min():.3f} ~ {test_output.max():.3f}")
