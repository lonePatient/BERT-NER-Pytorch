#encoding:Utf-8
class ProgressBar():
    def __init__(self,n_batch,
                 eval_name = 'acc',
                 loss_name = 'loss',
                 width=30
                 ):
        self.width = width
        self.loss_name = loss_name
        self.eval_name = eval_name
        self.n_batch = n_batch
        self.use = 'on_batch_end'

    def step(self,batch_idx,loss,acc,use_time):
        recv_per = int(100 * (batch_idx + 1) / self.n_batch)
        if recv_per >= 100:
            recv_per = 100
        # 只显示train数据结果
        show_bar = ('[%%-%ds]' % self.width) % (int(self.width * recv_per / 100) * ">")
        show_str = '\r[training] %d/%d %s -%.1fs/step- %s: %.4f- %s: %.4f'
        print(show_str % (
            batch_idx+1,self.n_batch,show_bar, use_time,self.loss_name, loss,self.eval_name, acc),end='')


