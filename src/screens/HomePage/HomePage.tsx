import { useEffect, useState } from "react";
import { ContactInfoSection } from "./sections/ContactInfoSection/ContactInfoSection";
import { InfoSection } from "./sections/InfoSection";
import { FAQSection } from "./sections/FAQSection";
import { CollectionSection } from "./sections/CollectionSection";
import { VideoSection } from "./sections/VideoSection";
import NewsSection from "./sections/NewsSection/NewsSction";
import { client } from "../../services";

export const HomePage = () => {
  const [heroImage, setHeroImage] = useState<string | null>(null);

  useEffect(() => {
    const fetchHeroImage = async () => {
      try {
        const response = await client.get("uz/api/v1/home/get_header/");
        if (response.data && response.data.image) {
          setHeroImage(response.data.image);
        }
      } catch (error) {
        console.error("Failed to fetch hero image:", error);
      }
    };

    fetchHeroImage();
  }, []);

  return (
    <div className="min-h-screen bg-[#FFFCE0] w-full">
      <div
        className="relative w-full h-[100vh] bg-cover bg-center"
        style={{ backgroundImage: heroImage ? `url(${heroImage})` : "none" }}
      ></div>

      <NewsSection />
      <InfoSection />
      <VideoSection />
      <FAQSection />
      <CollectionSection />
      <ContactInfoSection />
    </div>
  );
};
