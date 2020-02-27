# -*- coding: utf-8 -*-
# @Software: PyCharm
# @File: function_score.py
# @Author: MX
# @E-mail: minxinm@foxmail.com
# @Time: 2020/2/19

import os

import wx

from function_score import ScoreEngine


class mainFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent=parent, title='肽谱匹配打分', size=(550, 300))
        self.panel = wx.Panel(self)

        self.title = wx.StaticText(self.panel, -1, label='deepScore-α 打分工具', pos=(100, 10), size=(150, 20), style = wx.ALIGN_CENTER)
        font = wx.Font(18, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)
        self.title.SetFont(font)
        self.info = wx.StaticText(self.panel, -1, label='提示：需要进行打分的候选肽文件(.txt)以及相应质谱文件(.mgf)应存储在同一目录data文件目录下，文件格式参考测试用例',
                                  pos=(100, 50), size=(300, 60), style = wx.ALIGN_CENTER)
        self.text0 = wx.StaticText(self.panel, -1, label='NCE:', pos=(100, 150), size=(50, 20))
        self.text1 = wx.TextCtrl(self.panel, value='30', pos=(160, 150), size=(50, 20))

        self.btnT = wx.Button(self.panel, label="运行测试用例", pos=(300, 120), size=(100, 40))
        self.btnT.Bind(wx.EVT_BUTTON, self.CaculateScore_test)

        self.btnS = wx.Button(self.panel, label="开始打分计算", pos=(300, 160), size=(100, 40))
        self.btnS.Bind(wx.EVT_BUTTON, self.CaculateScore)

        self.Status = self.CreateStatusBar()
        self.Status.SetStatusText('--- 欢迎使用deepScore-α 打分工具 ---')
        self.Show(True)

    def CaculateScore(self, event):
        """
        根据选择的质谱谱图文件和候选肽存储文件计算分数输出，并提示是否计算FDR以及绘出FDR(q-value)曲线图
        :param event:
        :return:
        """
        spectrumfile = ''
        peptidefile = ''
        for f in os.listdir('./data'):
            if f not in ['test_peptide.txt', 'test_spectrum.mgf']:
                file_extention = os.path.splitext(f)[-1]
                if file_extention == '.txt':
                    peptidefile = f
                elif file_extention == '.mgf':
                    spectrumfile = f
        if spectrumfile == '':
            wx.MessageBox("请在data文件夹中放入测试质谱谱图文件", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        if peptidefile == '':
            wx.MessageBox("请在data文件夹中放入测试候选肽存储文件", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        nce = self.text1.GetValue()
        if nce not in ['30', '35']:
            wx.MessageBox("输入正确碰撞能量:30 or 35", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        scoreengine = ScoreEngine('./data/%s'%(peptidefile), './data/%s'%(spectrumfile), nce, self.Status)
        scoreengine.productPSMs()
        scoreengine.caculateScore()
        self.scoreengine = scoreengine
        self.GetFDR()

    def CaculateScore_test(self, event):
        """
        根据选择的质谱谱图文件和候选肽存储文件计算分数输出，并提示是否计算FDR以及绘出FDR(q-value)曲线图
        :param event:
        :return:
        """
        if not os.path.exists('./data/test_spectrum.mgf'):
            wx.MessageBox("请在data文件夹中放入测试质谱谱图文件", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        if not os.path.exists('./data/test_peptide.txt'):
            wx.MessageBox("请在data文件夹中放入测试候选肽存储文件", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        nce = self.text1.GetValue()
        if nce not in ['30', '35']:
            wx.MessageBox("输入正确碰撞能量:30 or 35", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        scoreengine = ScoreEngine('./data/test_peptide.txt', './data/test_spectrum.mgf', nce, self.Status)
        scoreengine.productPSMs()
        scoreengine.caculateScore()
        self.scoreengine = scoreengine
        self.GetFDR()

    def GetFDR(self):
        """
        分数计算成功后提示是否计算FDR并绘图，FDR文件及图片路径为当前目录
        :return:
        """
        self.msg1 = wx.MessageDialog(parent=None, message="分数计算成功，是否计算FDR并作出FDR(q-value)曲线图?", caption="提示消息",
                                     style=wx.YES_NO | wx.ICON_INFORMATION)
        # 如果选择“是”，startfile用来打开一个文件或者文件夹(像日常双击一样的效果)
        if self.msg1.ShowModal() == wx.ID_YES:
            self.scoreengine.caculateFDR_Plot()


if __name__ == '__main__':
    app = wx.App()
    frame = mainFrame(None)
    frame.Show()
    app.MainLoop()
