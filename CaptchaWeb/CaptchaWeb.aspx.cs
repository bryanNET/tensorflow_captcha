using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace CaptchaWeb
{
    public partial class CaptchaWeb : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            try
            {
                string code = Request.QueryString["code"];
                if (!string.IsNullOrWhiteSpace(code) && code.Trim().Length == 4)
                    dd(code.ToUpper());

                // Logs.AddLogs(code);
            }
            catch (Exception ex)
            {
                 //Logs.AddLogs(ex.Message);
                throw;
            }

        }

        private string[] fonts = { "Arial", "Helvetica", "Geneva", "sans-serif", "Verdana" };
        private Color[] colors = { Color.Black, Color.Red, Color.DarkBlue, Color.Green, Color.Orange, Color.Brown, Color.DarkCyan, Color.Purple };
        private Color chaosColor = Color.LightGray;
        private Color backgroundColor = Color.White;


        public void dd(string validateCode)
        {
            try
            {
                Response.Cache.SetCacheability(HttpCacheability.NoCache);
                string randomcode = validateCode;// Globals.CreateVerifyCode(4);

                // 随机转动角度33
                int randAngle = 45;
                int mapwidth = (int)(randomcode.Length * 50);
                // 创建图片背景- 5  45
                Bitmap map = new Bitmap(mapwidth, 80);
                Graphics graph = Graphics.FromImage(map);
                // 清除画面，填充背景
                //graph.Clear(Color.AliceBlue);
                graph.Clear(backgroundColor);
                // 画一个边框
                //graph.DrawRectangle(new Pen(Color.Gray, 0), 0, 0, map.Width-1, map.Height - 3);
                //graph.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;//模式

                Random rand = new Random();

                // 背景噪点生成
                //Pen blackPen = new Pen(Color.LightGray, 0);
                Pen blackPen = new Pen(chaosColor, 0);
                for (int i = 0; i < 40; i++)
                {
                    int x = rand.Next(0, map.Width);
                    int y = rand.Next(0, map.Height);
                    graph.DrawRectangle(blackPen, x, y, 1, 1);
                }

                // 验证码旋转，防止机器识别
                // 拆散字符串成单字符数组
                char[] chars = randomcode.ToCharArray();

                // 文字距中
                StringFormat format = new StringFormat(StringFormatFlags.NoClip);
                format.Alignment = StringAlignment.Center;
                format.LineAlignment = StringAlignment.Center;

                // 定义颜色
                //Color[] c = { Color.Black, Color.Red, Color.DarkBlue, Color.Green, Color.Brown, Color.DarkCyan, Color.Purple, Color.DarkGreen };
                // 定义字体
                //string[] font = { "Verdana", "Microsoft Sans Serif", "Comic Sans MS", "Arial","Lucida Sans Unicode", Rockwell, Batang ,Times New Roman,Bernard MT Condensed};
                int fSize = 60;
                int cindex = rand.Next(colors.Length - 1);// rand.Next(7);
                //cindex = rand.Next(colors.Length - 1);
                for (int i = 0; i < chars.Length; i++)
                {

                    int findex = rand.Next(4);


                    findex = rand.Next(fonts.Length - 1);
                    Font f = new Font(fonts[findex], fSize, FontStyle.Bold);
                    Brush b = new System.Drawing.SolidBrush(colors[cindex]);
                    // 字体样式(参数2为字体大小)
                    //Font f = new System.Drawing.Font("Microsoft Sans Serif", 60, System.Drawing.FontStyle.Bold);
                    //Brush b = new System.Drawing.SolidBrush(c[cindex]);

                    //Point dot = new Point(23, 15);  Point dot = new Point(38, 25);
                    Point dot = new Point(38, 35);
                    // 测试X坐标显示间距的
                    //graph.DrawString(dot.X.ToString(),fontstyle,new SolidBrush(Color.Black),10,150);
                    // 转动的度数
                    float angle = rand.Next(-randAngle, randAngle);

                    // 移动光标到指定位置
                    graph.TranslateTransform(dot.X, dot.Y);
                    graph.RotateTransform(angle);
                    graph.DrawString(chars[i].ToString(), f, b, 1, 10, format);
                    //graph.DrawString(chars[i].ToString(),fontstyle,new SolidBrush(Color.Blue),1,1,format);
                    // 转回去
                    graph.RotateTransform(-angle);
                    // 移动光标到指定位置
                    graph.TranslateTransform(3, -dot.Y);
                }
                // 标准随机码
                //graph.DrawString(randomcode,fontstyle,new SolidBrush(Color.Blue),2,2);

                // 生成图片
                System.IO.MemoryStream ms = new System.IO.MemoryStream();
                map.Save(ms, System.Drawing.Imaging.ImageFormat.Png);

                Response.ClearContent();
                Response.ContentType = "image/png";
                Response.BinaryWrite(ms.ToArray());

                graph.Dispose();
                map.Dispose();
            }
            catch
            {
            }

        }

    }
}