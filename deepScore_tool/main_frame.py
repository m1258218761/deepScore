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
        wx.Frame.__init__(self, parent=parent, title='肽谱匹配打分', size=(750, 500))
        self.panel = wx.Panel(self)

        ## 定义控件，绑定响应函数
        self.title = wx.StaticText(self.panel, -1, label='deepScore-α 打分工具')
        font = wx.Font(18, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)
        self.title.SetFont(font)
        self.info = wx.StaticText(self.panel, -1,
                                  label='提示：如需要对自定义的候选肽文件(.txt)进行打分，则文件格式应与测试用例一致',
                                  size=(500, 60), style=wx.ALIGN_CENTER)

        self.nce = '0'
        statictext_nce = wx.StaticText(self.panel, label='NCE: ')
        list_nce = ['30', '35']
        ch_nce = wx.ComboBox(self.panel, -1, value=' ', choices=list_nce, style=wx.CB_SORT)
        self.Bind(wx.EVT_COMBOBOX, self.choice_nce, ch_nce)

        self.ppm = '0'
        statictext_ppm = wx.StaticText(self.panel, label='离子匹配误差(ppm): ')
        list_ppm = ['10', '15', '20', '25', '30']
        ch_ppm = wx.ComboBox(self.panel, -1, value=' ', choices=list_ppm, style=wx.CB_SORT)
        self.Bind(wx.EVT_COMBOBOX, self.choice_ppm, ch_ppm)

        self.fdrlabel = wx.StaticText(self.panel, -1, label='鉴定结果FDR阈值(%):')
        self.fdrvalue = wx.TextCtrl(self.panel, value='1.00')

        self.flformat = 'txt'
        statictext_flformat = wx.StaticText(self.panel, label='输出文件格式: ')
        list_flformat = ['txt', 'csv', 'xlsx']
        ch_flformat = wx.ComboBox(self.panel, -1, value='txt', choices=list_flformat, style=wx.CB_SORT)
        self.Bind(wx.EVT_COMBOBOX, self.choice_flformat, ch_flformat)

        self.software = 'Comet'
        list_software = ['Comet', 'MSGF+', '自定义']
        self.ch_software = wx.RadioBox(self.panel, -1, "候选肽来源", choices=list_software, majorDimension=3,
                                       style=wx.RA_SPECIFY_COLS)
        self.ch_software.Bind(wx.EVT_RADIOBOX, self.choice_sofeware)

        self.addstates = {'isScorefile': False, 'isFdrplot': False}
        self.save_score = wx.CheckBox(self.panel, label='生成候选肽分数文件')
        self.fdr_plot = wx.CheckBox(self.panel, label='绘制FDR曲线图')
        self.Bind(wx.EVT_CHECKBOX, self.choice_state)

        self.btnS = wx.Button(self.panel, label="开始打分计算")
        self.btnS.Bind(wx.EVT_BUTTON, self.CaculateScore)

        ## 进行布局，添加控件
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.title, 0, wx.ALL | wx.CENTER, 5)
        vbox.Add(self.info, 0, wx.ALL | wx.CENTER, 5)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        scorebox = wx.StaticBox(self.panel, -1, '分数计算相关:')
        scoresizer = wx.StaticBoxSizer(scorebox, wx.VERTICAL)
        ncebox = wx.BoxSizer(wx.HORIZONTAL)
        ncebox.Add(statictext_nce, 0, wx.ALL | wx.CENTER, 5)
        ncebox.Add(ch_nce, 0, wx.ALL | wx.CENTER, 5)
        ppmbox = wx.BoxSizer(wx.HORIZONTAL)
        ppmbox.Add(statictext_ppm, 0, wx.ALL | wx.CENTER, 5)
        ppmbox.Add(ch_ppm, 0, wx.ALL | wx.CENTER, 5)
        scoresizer.Add(ncebox, 0, wx.ALL | wx.ALIGN_LEFT, 5)
        scoresizer.Add(ppmbox, 0, wx.ALL | wx.ALIGN_LEFT, 5)
        hbox.Add(scoresizer, 0, wx.ALL | wx.CENTER, 5)

        idtbox = wx.StaticBox(self.panel, -1, '鉴定结果相关:')
        idtsizer = wx.StaticBoxSizer(idtbox, wx.VERTICAL)
        fdrbox = wx.BoxSizer(wx.HORIZONTAL)
        fdrbox.Add(self.fdrlabel, 0, wx.ALL | wx.CENTER, 5)
        fdrbox.Add(self.fdrvalue, 0, wx.ALL | wx.CENTER, 5)
        ffbox = wx.BoxSizer(wx.HORIZONTAL)
        ffbox.Add(statictext_flformat, 0, wx.ALL | wx.CENTER, 5)
        ffbox.Add(ch_flformat, 0, wx.ALL | wx.CENTER, 5)
        idtsizer.Add(fdrbox, 0, wx.ALL | wx.ALIGN_LEFT, 5)
        idtsizer.Add(ffbox, 0, wx.ALL | wx.ALIGN_LEFT, 5)
        hbox.Add(idtsizer, 0, wx.ALL | wx.CENTER, 5)

        vbox.Add(hbox, 0, wx.ALL | wx.CENTER, 5)

        lbox = wx.BoxSizer(wx.HORIZONTAL)
        lbox.Add(self.ch_software, 0, wx.ALL | wx.CENTER, 5)
        lbox.Add(self.save_score, 0, wx.ALL | wx.CENTER, 5)
        lbox.Add(self.fdr_plot, 0, wx.ALL | wx.CENTER, 5)

        vbox.Add(lbox, 0, wx.ALL | wx.CENTER, 5)
        vbox.Add(self.btnS, 0, wx.ALL | wx.CENTER, 5)

        self.panel.SetSizer(vbox)

        self.Status = self.CreateStatusBar()
        self.Status.SetStatusText('欢迎使用deepScore-α 打分工具')
        self.Show(True)

    def choice_nce(self, event):
        """
        选择进行打分的碰撞能量(NCE)
        :param event:
        :return:
        """
        self.nce = event.GetString()
        print(self.nce)

    def choice_ppm(self, event):
        """
        选择碎片离子标注误差限(ppm)
        :param event:
        :return:
        """
        self.ppm = event.GetString()
        print(self.ppm)

    def choice_flformat(self, event):
        """
        选择最终输出的鉴定结果文件格式
        :param event:
        :return:
        """
        self.flformat = event.GetString()
        print(self.flformat)

    def choice_sofeware(self, event):
        """
        选择输入的候选肽来源：Comet、MSGF+以及自定义输入
        :param event:
        :return:
        """
        self.software = self.ch_software.GetStringSelection()
        print(self.software)

    def choice_state(self, event):
        """
        选择可执行的额外功能
        :param event:
        :return:
        """
        cb = event.GetEventObject()
        if cb.GetLabel() == '生成候选肽分数文件':
            self.addstates['isScorefile'] = cb.GetValue()
        else:
            self.addstates['isFdrplot'] = cb.GetValue()
        print(self.addstates)

    def CaculateScore(self, event):
        """
        根据质谱谱图文件和候选肽存储文件计算分数输出
        :param event:
        :return:
        """
        if self.software == 'MSGF+':
            peptidefile = 'msgf_output.tsv'
        elif self.software == 'Comet':
            peptidefile = 'comet_output.txt'
        else:
            peptidefile = 'customize_output.txt'
        spectrumfile = 'spectrum.mgf'
        if not os.path.exists('./data/' + spectrumfile):
            wx.MessageBox("请在data文件夹中放入质谱谱图文件", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        if not os.path.exists('./data/' + peptidefile):
            wx.MessageBox("请在data文件夹中放入候选肽存储文件", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        fdr = self.fdrvalue.GetValue()
        if not 0.0 < float(fdr) < 100.0:
            wx.MessageBox("输入正确FDR值", "提示消息", wx.OK | wx.YES_DEFAULT)
            return
        scoreengine = ScoreEngine('./data/%s' % (peptidefile), './data/%s' % (spectrumfile), self.nce, self.software, self.ppm, fdr, self.flformat, self.addstates, self.Status)
        scoreengine.productPSMs()
        allPsms_score = scoreengine.caculateScore()
        scoreengine.caculateFDR(allPsms_score)

if __name__ == '__main__':
    app = wx.App()
    frame = mainFrame(None)
    frame.Show()
    app.MainLoop()
