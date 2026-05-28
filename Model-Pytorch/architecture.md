flowchart TD
    %% 스타일 정의
    classDef inputBox fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef modelBox fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef moduleBox fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef lossBox fill:#ffebee,stroke:#b71c1c,stroke-width:2px;
    classDef evalBox fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;

    %% 1. Input State
    subgraph Input_State ["1. Input State (Step t)"]
        Prev_State["Prev State\n(p_t-1, v_t-1, a_t-1)"]:::inputBox
        Target_Char["Target Character\n(c_t)"]:::inputBox
    end

    %% 2. Main Generative Model
    subgraph Main_Model ["2. Main Generative Model"]
        LSTM_Main["Main LSTM Layers\n(Temporal Features)"]:::modelBox
        Dense_Main["Dense / Projection Layers"]:::modelBox
        GMM_Params["GMM Parameters\n(π, μ, σ, ρ, pen_state)"]:::modelBox
    end

    %% 3. Corrector Module
    subgraph Corrector_Module ["3. Corrector Module (one_step)"]
        Sensory_Feedback["Sensory Feedback\n(Target, Current Est.)"]:::moduleBox
        LSTM_Corrector["Corrector Network\n(LSTM or FFN)"]:::moduleBox
        Correction_Vector["Correction Vector\n(δ_t)"]:::moduleBox
    end

    %% 4. Sampling & Kinematics
    subgraph Sampling_Kinematics ["4. Sampling & Kinematics"]
        Sampler["GMM Sampler\n(sample_gmm_batch_pt)"]:::moduleBox
        Raw_Delta["Raw Δx, Δy\n(from Main Model)"]:::moduleBox
        Physics["Kinematics Engine\n(get_kinematics)"]:::moduleBox
        Final_State["Next State\n(p_t, v_t, a_t)"]:::inputBox
    end

    %% 5. Loss & Evaluation
    subgraph Loss_Eval ["5. Loss & Evaluation"]
        NLL_Loss["Main Loss\n(Negative Log-Likelihood)"]:::lossBox
        Corr_Loss["Corrector Loss\n(MSE or GMM)"]:::lossBox
        Total_Loss["Total Loss Calculation"]:::lossBox
        CNN_Eval["CNN Classifier (cnn.pt)\n*Epoch End Evaluation*"]:::evalBox
        Metrics["Metrics\n(Accuracy, Precision, Recall)"]:::evalBox
    end

    %% 흐름 연결 (Connections)
    Prev_State --> LSTM_Main
    Target_Char --> LSTM_Main
    
    LSTM_Main --> Dense_Main
    Dense_Main --> GMM_Params
    
    GMM_Params --> Sampler
    Sampler --> Raw_Delta
    
    Raw_Delta --> Sensory_Feedback
    Sensory_Feedback --> LSTM_Corrector
    LSTM_Corrector --> Correction_Vector
    
    Raw_Delta --> Physics
    Correction_Vector -->|"+ Added to Mean"| Physics
    
    Physics --> Final_State
    
    GMM_Params --> NLL_Loss
    Correction_Vector --> Corr_Loss
    NLL_Loss --> Total_Loss
    Corr_Loss --> Total_Loss
    
    Final_State -.->|Autoregressive Feedback: t+1| Prev_State
    
    Final_State -->|Synthetic Trajectories: Generated at Test Time| CNN_Eval
    CNN_Eval --> Metrics


graph TD
    %% 파일 및 모듈 단위
    subgraph train.py [1. Main Training Loop]
        TR_M[train]
    end

    subgraph model.py [2. Generative Model]
        MD_F[forward / infer_batch]
        MD_K[get_kinematics]
        MD_S[sample_gmm_batch_pt]
        
        subgraph Corrector Class
            CR_F[forward]
            CR_O[one_step]
        end
    end

    subgraph logger.py [3. Evaluation & Logging]
        LG_T[log_test_synth]
    end

    subgraph cnn_classifier.py [4. Evaluator]
        CNN_C[classify]
        CNN_M((cnn.pt))
    end

    %% 데이터 흐름 및 함수 호출 관계
    TR_M -- 1. 학습/추론 지시 --> MD_F
    
    MD_F -- 2. 궤적 샘플링 --> MD_S
    MD_S -- 3. 다음 좌표/GMM 예측 반환 --> MD_F
    
    MD_F -- 4. 보정 지시 --> CR_O
    CR_O -- 5. 신경망(LSTM/FFN) 계산 --> CR_F
    CR_F -- 6. 보정된 좌표 반환 --> CR_O
    CR_O -- 7. 최종 궤적 반환 --> MD_F
    
    MD_F -- 8. 물리적 제약/정규화 적용 --> MD_K
    MD_K -- 9. 궤적 완성 --> TR_M
    
    TR_M -- 10. Epoch 종료 시 평가 지시 --> LG_T
    LG_T -- 11. 완성된 궤적 데이터 전달 --> CNN_C
    
    CNN_C -- 12. 디바이스 동기화 및 예측 --> CNN_M
    CNN_M -- 13. 분류 결과 (숫자 0~9) --> CNN_C
    CNN_C -- 14. Accuracy, F1 Score 반환 --> LG_T