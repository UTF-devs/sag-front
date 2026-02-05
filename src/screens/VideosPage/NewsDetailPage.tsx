import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Eye,
  Clock,
  ArrowLeft,
} from "lucide-react";
import { ContactInfoSection } from "../HomePage/sections/ContactInfoSection";
import { useLanguage } from "../../contexts/LanguageContext";
import { client } from "../../services";

interface SocialLink {
  id: number;
  link: string;
  icon: {
    name: string;
    image: string;
  };
}

interface NewsData {
  id: number;
  title: string;
  description: string;
  image: string;
  view_count: number;
  links: SocialLink[];
  created_at: string;
}

const NewsDetail = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { t, language } = useLanguage();
  const [news, setNews] = useState<NewsData | null>(null);

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

  useEffect(() => {
    const fetchNewsDetail = async () => {
      try {
        const lang = mapLang(language);
        const res = await client.get(`/${lang}/api/v1/news/get_news_by_id/${id}/`);
        setNews(res.data);
      } catch (error) {
        console.error("Error fetching news detail:", error);
      }
    };

    if (id) {
      fetchNewsDetail();
    }
  }, [id, language]);

  if (!news) return null; 

  return (
    <div className="md:pt-28 pt-24 bg-[#FFFCE0] min-h-screen flex flex-col">
      <div className="container mx-auto px-4 py-8 flex-1">
        <button
          onClick={() => navigate("/methods")}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-800 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          <span className="md:text-2xl">{t("video.back")}</span>
        </button>

        <h1 className="md:text-3xl text-xl font-semibold text-gray-800 mb-6">
          {news.title}
        </h1>

        <div className="mb-4">
          <img
            src={news.image}
            alt={news.title}
            className="w-full h-[500px] md:h-[800px] object-cover rounded-lg"
            loading="lazy"
          />
        </div>

        <div className="flex flex-wrap items-center justify-between text-[#8A6E4F] text-sm mb-6">
          <div className="flex items-center space-x-4">
            <div className="flex items-center gap-1">
              <Clock size={16} />
              <span>{news.created_at?.slice(0, 10)}</span>
            </div>
            <div className="flex items-center gap-1">
              <Eye size={16} />
              <span>{news.view_count}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {Array.isArray(news.links) &&
              news.links.map((item) => (
                <a
                  key={item.id}
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:opacity-75 transition"
                  title={item.icon.name}
                >
                  <img
                    src={item.icon.image}
                    alt={item.icon.name}
                    className="w-5 h-5 object-contain"
                  />
                </a>
              ))}
          </div>
        </div>

        <p className="text-gray-700 text-base">{news.description}</p>
      </div>

      <ContactInfoSection />
    </div>
  );
};

export default NewsDetail;
