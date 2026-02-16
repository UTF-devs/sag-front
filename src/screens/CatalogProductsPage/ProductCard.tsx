import { Link } from "react-router-dom";
import { useLanguage } from "../../contexts/LanguageContext";
import { useState } from "react";

interface ProductCardProps {
  id: number;
  name: string;
  image: string;
  collection_type: string;
  price?: number;
  onClick?: () => void;
}

const ProductCard = ({
  id,
  name,
  image,
  collection_type,
  price,
  onClick,
}: ProductCardProps) => {
  const { t } = useLanguage();
  const [showPrice, setShowPrice] = useState(false);

  const handlePriceToggle = () => {
    setShowPrice(!showPrice);
  };

  const getCollectionTypeLabel = (type?: string) => {
    if (!type) return "";
    switch (type.toLowerCase()) {
      case "new":
        return t("badge.new") || "Yangi";
      case "sale":
        return t("badge.sale") || "Chegirma";
      case "hit":
        return t("hit") || "Mashhur";
      default:
        return "";
    }
  };

  const getBadgeColor = (type?: string) => {
    if (!type) return "border-t-gray-500";
    switch (type.toLowerCase()) {
      case "new":
        return "border-t-green-600";
      case "sale":
        return "border-t-red-500";
      case "hit":
        return "border-t-orange-500";
      default:
        return "border-t-gray-500";
    }
  };

  return (
    <div className="shadow-md border-gray-200 overflow-hidden transition-shadow hover:shadow-md">
      <Link to={`/product/${id}`} className="group block">
        <div className="relative h-[200px] sm:h-[360px] md:h-[430px] bg-gray-100">
          <img
            src={image}
            alt={name}
            className="w-full h-full object-cover transform transition-transform duration-300 group-hover:scale-105"
          />

          {getCollectionTypeLabel(collection_type) && (
            <div className="absolute top-0 right-0 w-[30px] h-[30px] sm:w-[60px] sm:h-[60px]">
              <div className="absolute top-0 right-0 w-0 h-0 border-t-[60px] border-t-[#FFFCE0] border-l-[60px] border-l-transparent z-10 sm:border-t-[100px] sm:border-l-[100px]" />
              <div
                className={`absolute top-0 right-0 w-0 h-0 border-t-[60px] border-l-[60px] border-l-transparent z-20 rotate-180 sm:border-t-[100px] sm:border-l-[100px] ${getBadgeColor(collection_type)}`}
              />
              <span className="absolute top-[38px] right-[19px] text-white text-[8px] font-bold z-30 sm:top-[70px] sm:right-[37px] sm:text-[13px]">
                {getCollectionTypeLabel(collection_type)}
              </span>
            </div>
          )}
        </div>

        <div className="p-4">
          <h3 className="font-medium text-gray-800 mb-2 group-hover:text-gray-500 transition-colors">
            {name}
          </h3>
        </div>
      </Link>

      <div className="px-4 pb-4">
        {price !== undefined ? (
          showPrice ? (
            <button
              className="font-semibold border border-black text-black w-full py-2 rounded"
              onClick={handlePriceToggle}
            >
              {price.toLocaleString()} {t("currency")}
            </button>
          ) : (
            <button
              className="bg-transparent border border-black text-black w-full py-2 rounded"
              onClick={handlePriceToggle}
            >
              {t("show_price")}
            </button>
          )
        ) : (
          <button className="bg-transparent border border-black text-black w-full py-2 rounded">
            0
          </button>
        )}
      </div>
    </div>
  );
};

export default ProductCard;
