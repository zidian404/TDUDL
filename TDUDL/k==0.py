 if k == 0:
                # update x
                X_in = X
                X1 = self.D_0(X_in)
                temp = torch.sub(self.D_0T(X1), input)
                X2 = self.D_0(temp)
                X_ = X2 + torch.mul(rho, X1)
                X = X1 - torch.mul(alpha, X_)
                # update z
                rho_ = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(torch.cat([X, rho_], dim=1), stage_inter=True)# torch.cat([X, rho_], dim=1),
                # Z, samfeats, enc, dec = self.unet(X, rho)
                # update beta
                beta = gamma[1] * X - gamma[2] * Z
                
                # preds
                output = self.D(X)
                preds.append(output)


                if k == 0:
                X_in = X 
                X1_img = self.S(X_in) + apply_Di(X_in, Di_batch) 
                X1_back = self.S_T(X1_img) + apply_Di_T(X1_img, Di_batch)
                input_back = self.S_T(input) + apply_Di_T(input, Di_batch)
                temp = X1_back - input_back
                X2 = self.S(temp) + apply_Di(temp, Di_batch) 
                X_ = X2 + torch.mul(rho, X1_img)
                X = X1_img - torch.mul(alpha, X_)

                # --- 2. Update Z (对应原代码 update z) ---
                rho_ = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                # 这里的 X 已经是更新后的系数域变量
                Z, samfeats, enc, dec = self.unet(torch.cat([X, rho_], dim=1), stage_inter=True)

                # --- 3. Update Beta (对应原代码 update beta) ---
                # beta = gamma[1] * X - gamma[2] * Z
                beta = gamma[1] * X - gamma[2] * Z
                
                # --- 4. Preds (对应原代码 preds) ---
                # output = self.D(X) 
                # 对应新逻辑: 最终重构输出到图像域
                output = self.S(X) + apply_Di(X, Di_batch)
                preds.append(output)