import unittest

from remote_chatglm import RemoteChatGLM


class RemoteChatGLMTest(unittest.TestCase):
    def test_basic(self):
        remote_chatglm = RemoteChatGLM()
        self.assertEquals("http://chatglm.nigelliang.com:8000", remote_chatglm.remote_host)
        self.assertIn("RemoteChatGLM", str(remote_chatglm))

        remote_chatglm = RemoteChatGLM(remote_host="another remote host")
        self.assertEquals("another remote host", remote_chatglm.remote_host)

    def test_hit_host(self):
        remote_chatglm = RemoteChatGLM()
        resp = remote_chatglm("你好")
        self.assertEquals("你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。", resp)


if __name__ == '__main__':
    unittest.main()
