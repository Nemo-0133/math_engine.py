import numpy as np

class QuantumStateEngine:
    def __init__(self, num_branches=3):
        self.num_branches = num_branches
        # 初始化疊加態振幅 (0.5 ~ 1.0 之間)
        self.state = np.random.uniform(0.5, 1.0, num_branches)
        self.dt = 0.1  # 時間步長 (Time step)

    def step(self, base_lambda, gamma):
        """
        執行單步量子狀態推演 (Euler-Maruyama 方法)
        """
        # 1. 計算當前狀態的資訊熵 (用變異數的倒數作為代理指標)
        variance = np.var(self.state) + 1e-6
        entropy = 1.0 / (variance * 100)
        
        # 2. 計算動態衰變常數: 熵越高，衰變越慢 (保護疊加態)
        actual_lambda = base_lambda / (entropy + 1.0)
        
        # 3. 生成量子噪聲 (維納過程 / 標準布朗運動)
        dW = np.random.normal(0, np.sqrt(self.dt), self.num_branches)
        
        # 4. SDE 演化公式: dX = -\lambda * X * dt + \sigma * dW
        # 觀測係數 gamma 會放大或縮小隨機擾動 (sigma)
        sigma = 0.05 * gamma
        drift = -actual_lambda * self.state * self.dt
        diffusion = sigma * dW
        
        self.state += drift + diffusion
        
        # 將機率約束在 [0, 1] 之間
        self.state = np.clip(self.state, 0.01, 0.99)
        
        # 5. 齊諾坍縮判定 (當 gamma 突破 1.4，或某分支機率超過 0.95 時觸發)
        collapsed = False
        collapse_msg = ""
        if gamma > 1.4 or np.max(self.state) > 0.95:
            collapsed = True
            winner_idx = np.argmax(self.state)
            # 贏者全拿 (Winner-takes-all)
            self.state = np.full(self.num_branches, 0.01)
            self.state[winner_idx] = 1.0
            collapse_msg = f">> 觸發齊諾觀測！分支 Ψ_{winner_idx} 坍縮確立 <<"
            
        return self.state.tolist(), actual_lambda, collapsed, collapse_msg
