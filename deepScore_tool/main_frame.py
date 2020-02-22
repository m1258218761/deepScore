# -*- coding: utf-8 -*-
# @Software: PyCharm
# @File: function_score.py
# @Author: MX
# @E-mail: minxinm@foxmail.com
# @Time: 2020/2/19

import os
import re
import sys

import wx
import win32

from function_score import ScoreEngine


class mainFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent=parent, title='deepScore-α 打分工具', size=(550, 200))
        self.panel = wx.Panel(self)

        self.text1 = wx.TextCtrl(self.panel, value='30', pos=(30, 20), size=(200, 25))

        self.btnS = wx.Button(self.panel, label="打分计算", pos=(450, 35), size=(70, 35))
        self.btnS.Bind(wx.EVT_BUTTON, self.CaculateScore)

    def CaculateScore(self, event):
        """
        根据选择的质谱谱图文件和候选肽存储文件计算分数输出，并提示是否计算FDR以及绘出FDR(q-value)曲线图
        :param event:
        :return:
        """

        if not os.path.exists('./data/test_spectrum.mgf') :
            wx.MessageBox("请在data文件夹中放入质谱谱图文件", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        if not os.path.exists('./data/test_peptide.txt') :
            wx.MessageBox("请在data文件夹中放入候选肽存储文件", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        nce = self.text1.GetValue()
        if nce not in ['30', '35']:
            wx.MessageBox("输入正确碰撞能量:30 or 35", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        scoreengine = ScoreEngine('./data/test_peptide.txt', './data/test_spectrum.mgf', nce)
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